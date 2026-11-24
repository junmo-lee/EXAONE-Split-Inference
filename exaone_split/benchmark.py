"""Run ordered inference phases and preserve each completed measurement."""
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from .client import encode, generate, run_tcp

CONDITIONS = [(0, 0), (0.2, 0), (1, 0), (5, 0), (20, 0), (1, 1000), (1, 10000)]


def measure(run):
    import shutil

    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            data = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            )
            if data.returncode == 0:
                try:
                    samples.append([int(value) for value in data.stdout.strip().split(",")])
                except ValueError:
                    pass
            stop.wait(0.15)

    worker = threading.Thread(target=sample) if shutil.which("nvidia-smi") else None
    if worker:
        worker.start()
    try:
        result = run()
    finally:
        stop.set()
        if worker:
            worker.join()
    result["gpu_samples"] = [
        dict(vram_mib=memory, utilization_percent=utilization) for memory, utilization in samples
    ]
    return result


def official(model_dir, prompt, max_tokens, repetitions, device="cuda:0"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    kwargs = {}
    if device.startswith("cuda"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, device_map={"": device}, dtype=torch.bfloat16, **kwargs,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = encode(tokenizer, prompt).to(device)

    first_logits = []

    def forward(num_tokens):
        steps, produced = [], []
        cache = None
        current = ids
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for step in range(num_tokens):
                begin = time.perf_counter()
                out = model(input_ids=current, attention_mask=torch.ones(
                    (ids.shape[0], ids.shape[1] + step), dtype=torch.long, device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                forward_seconds = time.perf_counter() - begin
                token = out.logits[:, -1].argmax(dim=-1).item()
                steps.append(dict(step=step, gpu_seconds=forward_seconds if device.startswith("cuda") else None,
                                  total_seconds=time.perf_counter() - begin))
                if step == 0:
                    first_logits.append(out.logits[:, -1].float().cpu())
                cache = out.past_key_values
                produced.append(token)
                if token == tokenizer.eos_token_id:
                    break
                current = ids.new_tensor([[token]])
        decode_seconds = sum(s["total_seconds"] for s in steps[1:])
        report = dict(prompt_tokens=ids.shape[1], generated_tokens=len(produced), token_ids=produced,
                      output=tokenizer.decode(produced, skip_special_tokens=True),
                      prefill_seconds=steps[0].get("gpu_seconds") or steps[0]["total_seconds"], first_token_seconds=steps[0]["total_seconds"],
                      decode_tokens_per_second=(len(produced) - 1) / decode_seconds if decode_seconds else None,
                      client_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      steps=steps)
        if device.startswith("cuda"):
            report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return report

    forward(min(max_tokens, 2))
    return [measure(lambda: forward(max_tokens)) for _ in range(repetitions)], first_logits[0]


def run_benchmark(args):
    import gc
    import socket
    import torch
    from .model import CpuFront, SplitShard
    from .server import gpu_forward

    torch.set_num_threads(args.threads)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(prompt=args.prompt, threads=args.threads, repetitions=args.repetitions,
                  runs=[], failures=[], summaries=[])

    def save():
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        pending.replace(output)

    def record(mode, split, rtt, bandwidth, repeat, result):
        result.update(mode=mode, split=split, rtt_ms=rtt, bandwidth_mbps=bandwidth, repeat=repeat)
        report["runs"].append(result)
        save()
        print(f"{mode} split={split} RTT={rtt} bandwidth={bandwidth} repeat={repeat}: "
              f"{result['decode_tokens_per_second']} tokens/s", flush=True)

    def fail(mode, exc):
        report["failures"].append(dict(mode=mode, error=f"{type(exc).__name__}: {exc}"))
        save()
        print(report["failures"][-1], flush=True)

    reference_logits = None
    if not args.skip_official:
        for device in ("cuda:0", "cpu"):
            try:
                runs, reference_logits = official(
                    args.dir, args.prompt, args.max_tokens, args.repetitions, device
                )
                for repeat, result in enumerate(runs):
                    record(f"official-{device}", None, None, None, repeat, result)
                break
            except Exception as exc:
                fail(f"official-{device}", exc)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    def local(front, ids, split):
        back = SplitShard(args.dir, split, 64, "cuda:0", quantize=True)
        hidden = front.embed_and_run(ids, 0)
        remote, _ = gpu_forward(back, hidden, 0)
        logits = front.logits(remote)
        if reference_logits is not None:
            diff = (logits - reference_logits).abs()
            report["prefill_logits_comparison"] = dict(
                mean_absolute_error=diff.mean().item(), max_absolute_error=diff.max().item(),
                matching_argmax=logits.argmax().item() == reference_logits.argmax().item(),
                official_gpu_all_layers_nf4_vs_split_cpu_layers_bf16=(
                    report["runs"][0]["mode"] == "official-cuda:0"
                ),
            )
        back.reset()
        generate(front, ids, 2, lambda x, pos: gpu_forward(back, x, pos))
        for repeat in range(args.repetitions):
            back.reset()
            torch.cuda.reset_peak_memory_stats()
            result = measure(lambda: generate(front, ids, args.max_tokens,
                                              lambda x, pos: gpu_forward(back, x, pos)))
            record("local", split, 0, 0, repeat, result)

    for split in args.splits:
        front = CpuFront(args.dir, split)
        ids = encode(front.tokenizer, args.prompt)
        if split == 4:
            try:
                local(front, ids, split)
            except Exception as exc:
                fail("local", exc)
            gc.collect()
            torch.cuda.empty_cache()
        log = output.with_name(f"{output.stem}-server-{split}.log")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with log.open("w") as sink:
            server = subprocess.Popen(
                [sys.executable, "-m", "exaone_split.cli", "server", "--dir", args.dir,
                 "--split", str(split), "--port", str(port)],
                stdout=sink, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(args.threads)},
            )
            try:
                deadline = time.monotonic() + 1200
                while "READY " not in log.read_text():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"GPU server did not start. See {log}")
                    time.sleep(0.25)
                conditions = CONDITIONS if split == 4 else [(1, 0)]
                run_tcp(front, ids, "127.0.0.1", port, split, 1, 0, 2)
                for rtt, bandwidth in conditions:
                    for repeat in range(args.repetitions):
                        result = measure(lambda: run_tcp(front, ids, "127.0.0.1", port, split,
                                                         rtt, bandwidth, args.max_tokens))
                        local_reference = next(
                            (r for r in report["runs"] if r["mode"] == "local" and r["split"] == split),
                            None,
                        )
                        if local_reference:
                            result["matches_local_token_ids"] = result["token_ids"] == local_reference["token_ids"]
                        record("tcp", split, rtt, bandwidth, repeat, result)
            except Exception as exc:
                fail(f"tcp-{split}", exc)
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    groups = {}
    for result in report["runs"]:
        key = tuple(result[k] for k in ("mode", "split", "rtt_ms", "bandwidth_mbps"))
        groups.setdefault(key, []).append(result)
    for key, runs in groups.items():
        summary = dict(zip(("mode", "split", "rtt_ms", "bandwidth_mbps"), key))
        summary["count"] = len(runs)
        for field in ("prefill_seconds", "first_token_seconds", "decode_tokens_per_second"):
            values = [r[field] for r in runs if r[field] is not None]
            if values:
                summary[field] = dict(mean=statistics.mean(values),
                                      sample_stdev=statistics.stdev(values) if len(values) > 1 else None)
        report["summaries"].append(summary)
    save()
