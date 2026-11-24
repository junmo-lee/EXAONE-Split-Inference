"""CPU-side greedy decoding and per-token measurements."""
import json
import resource
import socket
import time
from pathlib import Path

from .protocol import SimulatedLink, pack_hidden, recv_message, send_message, unpack_hidden


def encode(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=False, enable_thinking=False,
    )


def generate(front, ids, max_tokens, forward):
    import torch

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    initial_tokens = ids.shape[1]
    front.shard.reset()
    position = 0
    records, produced = [], []
    for step in range(max_tokens):
        begin = time.perf_counter()
        x = front.embed_and_run(ids, position)
        front_end = time.perf_counter()
        out, metrics = forward(x, position)
        head_begin = time.perf_counter()
        token = front.next_token(out)
        end = time.perf_counter()
        records.append(dict(step=step, input_length=ids.shape[1],
                            cpu_seconds=front_end - begin + end - head_begin,
                            forward_seconds=head_begin - begin,
                            head_seconds=end - head_begin, total_seconds=end - begin, **metrics))
        produced.append(token)
        position += ids.shape[1]
        if token == front.tokenizer.eos_token_id:
            break
        ids = torch.tensor([[token]], dtype=torch.long)
    elapsed = sum(r["total_seconds"] for r in records)
    decode_seconds = sum(r["total_seconds"] for r in records[1:])
    return dict(
        prompt_tokens=initial_tokens, generated_tokens=len(produced), token_ids=produced,
        elapsed_seconds=elapsed, tokens_per_second=len(produced) / elapsed,
        prefill_seconds=records[0]["forward_seconds"],
        first_token_seconds=records[0]["total_seconds"],
        decode_tokens_per_second=(len(produced) - 1) / decode_seconds if decode_seconds else None,
        bytes_sent=sum(r.get("bytes_sent", 0) for r in records),
        bytes_received=sum(r.get("bytes_received", 0) for r in records),
        client_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output=front.tokenizer.decode(produced, skip_special_tokens=True), steps=records,
    )


def run_tcp(front, ids, host, port, split, rtt_ms, bandwidth_mbps, max_tokens):
    link = SimulatedLink(rtt_ms, bandwidth_mbps)
    with socket.create_connection((host, port), timeout=30) as sock:
        sock.settimeout(180)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_message(sock, {"op": "reset", "split": split})
        reply, _, _ = recv_message(sock)
        if reply.get("op") != "ok":
            raise RuntimeError(f"GPU reset failed: {reply}")

        def forward(x, position):
            wire = pack_hidden(x)
            begin = time.perf_counter()
            sent = send_message(sock, {"op": "forward", "position": position}, wire)
            response, out, received = recv_message(sock)
            rpc = time.perf_counter() - begin
            if response.pop("op", None) != "output":
                raise RuntimeError(f"GPU forward failed: {response}")
            injected = link.delay(sent, received)
            return unpack_hidden(out), dict(
                rpc_seconds=rpc, injected_seconds=injected,
                tcp_and_framing_seconds=rpc - response["gpu_seconds"] - response["transfer_seconds"],
                bytes_sent=sent, bytes_received=received, **response,
            )

        return generate(front, ids, max_tokens, forward)


def run(model_dir, host="127.0.0.1", port=9876, split=4, rtt_ms=1.0,
        bandwidth_mbps=0.0, prompt="대한민국의 수도는 어디인가요?", max_tokens=16,
        output_json=None):
    from .model import CpuFront

    front = CpuFront(model_dir, split)
    ids = encode(front.tokenizer, prompt)
    report = run_tcp(front, ids, host, port, split, rtt_ms, bandwidth_mbps, max_tokens)
    report.update(model="EXAONE-4.5-33B", split=split, rtt_ms=rtt_ms,
                  bandwidth_mbps=bandwidth_mbps)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if output_json:
        Path(output_json).write_text(result)
    return report
