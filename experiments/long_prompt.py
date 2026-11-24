"""Compare two inference paths using a deterministic 4096-token synthetic prompt."""
import argparse
import gc
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from exaone_split.benchmark import measure, official
from exaone_split.client import encode, run_tcp
from exaone_split.model import CpuFront

TARGET = 4096
LINE = "The public test log records a blue sensor, a green marker, and a reading of 17.\n"
QUESTION = "\nFinal record: the access color is amber. What is the access color? Answer in one short sentence."


def make_prompt(tokenizer):
    prefix = "Read this synthetic log and answer the question at the end.\n"
    count = 0
    while encode(tokenizer, prefix + LINE * (count + 1) + QUESTION).shape[1] <= TARGET:
        count += 1
    text = prefix + LINE * count
    for padding in range(128):
        prompt = text + " x" * padding + QUESTION
        if encode(tokenizer, prompt).shape[1] == TARGET:
            return prompt
    raise ValueError("Could not construct an exact 4096-token prompt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-gpu-results", type=Path)
    parser.add_argument("--tcp-repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--split", type=int, choices=range(64), default=4)
    parser.add_argument("--short-tcp-warmup", action="store_true")
    args = parser.parse_args()
    if min(args.tcp_repetitions, args.threads) < 1:
        parser.error("Thread and repetition counts must be positive")
    if args.reuse_gpu_results and args.output.resolve() == args.reuse_gpu_results.resolve():
        parser.error("Use a different output path when reusing GPU results")
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.dir)
    prompt = make_prompt(tokenizer)
    data = dict(prompt=prompt, prompt_tokens=TARGET, tcp_cpu_threads=args.threads,
                gpu_only_host_threads=args.threads, max_new_tokens=12, repetitions=dict(gpu_only=3, tcp=args.tcp_repetitions),
                tcp_warmup_tokens=None, server_idle_timeout_seconds=600,
                split=args.split, rtt_ms=1, bandwidth_mbps=10000, runs=[], failures=[], summaries={})

    def save():
        pending = args.output.with_suffix(args.output.suffix + ".tmp")
        pending.write_text(json.dumps(data, indent=2))
        pending.replace(args.output)

    def record(mode, result):
        data["runs"].append(dict(mode=mode, **result))
        save()
        print(mode, result["first_token_seconds"], result["decode_tokens_per_second"],
              repr(result["output"]), flush=True)

    save()
    print(f"Starting GPU-only NF4 for {TARGET} tokens", flush=True)
    try:
        if args.reuse_gpu_results:
            previous = json.loads(args.reuse_gpu_results.read_text())
            data["gpu_only_host_threads"] = previous.get("gpu_only_host_threads", previous.get("threads"))
            if previous["prompt"] != prompt:
                raise ValueError("GPU results used a different prompt")
            runs = [{k: v for k, v in run.items() if k != "mode"} for run in previous["runs"]
                    if run["mode"] == "gpu-only-nf4"]
            if len(runs) != 3:
                raise ValueError("Expected three saved GPU-only runs")
        else:
            runs, _ = official(args.dir, prompt, 12, 3)
        for run in runs:
            record("gpu-only-nf4", run)
    except Exception as exc:
        data["failures"].append(dict(mode="gpu-only-nf4", error=str(exc)))
        save()
        print(type(exc).__name__, str(exc), flush=True)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    front = CpuFront(args.dir, args.split)
    ids = encode(front.tokenizer, prompt)
    assert ids.shape[1] == TARGET
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server_log = args.output.with_suffix(".server.log")
    with server_log.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "exaone_split.cli", "server", "--dir", args.dir,
             "--port", str(port), "--split", str(args.split), "--idle-timeout", "600"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            while "READY " not in server_log.read_text():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"GPU worker startup failed. See {server_log}")
                time.sleep(0.25)
            print("TCP split warmup", flush=True)
            warmup = encode(front.tokenizer, QUESTION) if args.short_tcp_warmup else ids
            data["tcp_warmup_tokens"] = warmup.shape[1]
            run_tcp(front, warmup, "127.0.0.1", port, args.split, 1, 10000, 2)
            for i in range(args.tcp_repetitions):
                print(f"TCP repeat {i + 1}/{args.tcp_repetitions}", flush=True)
                result = measure(lambda: run_tcp(front, ids, "127.0.0.1", port, args.split, 1, 10000, 12))
                record("tcp-split", result)
        except Exception as exc:
            data["failures"].append(dict(mode="tcp-split", error=str(exc)))
            save()
            raise
        finally:
            server.terminate()
            server.wait(timeout=15)

    for mode in ("gpu-only-nf4", "tcp-split"):
        runs = [run for run in data["runs"] if run["mode"] == mode]
        if not runs:
            continue
        summary = {}
        for key in ("prefill_seconds", "first_token_seconds", "decode_tokens_per_second"):
            vals = [r[key] for r in runs if r[key] is not None]
            if vals:
                summary[key] = dict(mean=statistics.mean(vals),
                                    sample_stdev=statistics.stdev(vals) if len(vals) > 1 else None)
        data["summaries"][mode] = summary
    save()


if __name__ == "__main__":
    main()
