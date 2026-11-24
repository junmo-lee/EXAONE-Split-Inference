import argparse


def main():
    p = argparse.ArgumentParser(description="EXAONE split inference lab (public data only)")
    sub = p.add_subparsers(dest="mode", required=True)
    fetch = sub.add_parser("fetch", help="Download the official sharded BF16 checkpoint")
    fetch.add_argument("--dir", required=True)
    verify = sub.add_parser("verify", help="Inspect local model metadata, no torch/GPU necessary")
    verify.add_argument("--dir", required=True)
    server = sub.add_parser("server")
    client = sub.add_parser("client")
    for cmd in (server, client):
        cmd.add_argument("--dir", required=True)
        cmd.add_argument("--host", default="127.0.0.1")
        cmd.add_argument("--port", type=int, default=9876)
        cmd.add_argument("--split", type=int, default=4)
    server.add_argument("--no-quant", action="store_true")
    server.add_argument("--idle-timeout", type=float, default=180,
                        help="Seconds to wait for the next request")
    client.add_argument("--rtt-ms", type=float, default=1.0)
    client.add_argument("--bandwidth-mbps", type=float, default=0.0)
    client.add_argument("--prompt", default="대한민국의 수도는 어디인가요?")
    client.add_argument("--max-tokens", type=int, default=16)
    client.add_argument("--output-json")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--dir", required=True)
    bench.add_argument("--output-json", required=True)
    bench.add_argument("--splits", type=int, nargs="+", default=[4, 2, 8, 16])
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--threads", type=int, default=8)
    bench.add_argument("--prompt", default="대한민국의 수도는 어디인가요? 한 문장으로 답하세요.")
    bench.add_argument("--max-tokens", type=int, default=12)
    bench.add_argument("--skip-official", action="store_true")
    args = p.parse_args()
    if args.mode == "benchmark":
        if min(args.threads, args.repetitions, args.max_tokens) < 1 or any(
            not 0 <= split < 64 for split in args.splits
        ):
            p.error("Invalid thread/repetition/token count or split")
        from .benchmark import run_benchmark

        run_benchmark(args)
        return
    if args.mode == "fetch":
        from .checkpoint import fetch_model
        print(fetch_model(local_dir=args.dir))
    elif args.mode == "verify":
        from .checkpoint import Checkpoint, PREFIX
        cp = Checkpoint(args.dir)
        count = sum(k.startswith(PREFIX + "layers.") for k in cp.key_map)
        assert count > 100, "No text decoder layer tensors found"
        from pathlib import Path
        import json
        cfg = json.loads((Path(args.dir)/"config.json").read_text())
        tc = cfg["text_config"]
        print({"model_type": cfg["model_type"], "text_model_type":tc["model_type"],
               "layers":tc["num_hidden_layers"], "hidden_size":tc["hidden_size"],
               "decoder_tensor_count":count,
               "quantization":"BF16 official checkpoint; NF4 runtime quantization on GPU shard"})
    elif args.mode == "server":
        from .server import serve
        serve(args.dir, args.host, args.port, args.split, not args.no_quant, args.idle_timeout)
    elif args.mode == "client":
        from .client import run
        run(args.dir, args.host, args.port, args.split, args.rtt_ms,
            args.bandwidth_mbps, args.prompt, args.max_tokens, args.output_json)


if __name__ == "__main__":
    main()
