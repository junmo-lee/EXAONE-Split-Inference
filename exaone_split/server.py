"""Single-client GPU shard for a trusted experimental link."""
import math
import resource
import socket
import time
from pathlib import Path

from .protocol import pack_hidden, recv_message, send_message, unpack_hidden


def process_peak_rss_kib():
    # Linux getrusage can retain the parent's high-water mark across fork/exec.
    try:
        for row in Path("/proc/self/status").read_text().splitlines():
            if row.startswith("VmHWM:"):
                return int(row.split()[1])
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def gpu_forward(shard, tensor, position, head=None):
    import torch

    begin = time.perf_counter()
    x = tensor.to(shard.device)
    transfer_in = time.perf_counter() - begin
    begin = time.perf_counter()
    out = shard.forward(x, position)
    if head is not None:
        out = head.next_token(out)
    torch.cuda.synchronize(shard.device)
    gpu_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    if head is None:
        out = out[:, -1:, :].to("cpu")
    return out, dict(
        gpu_seconds=gpu_seconds,
        transfer_seconds=transfer_in + time.perf_counter() - begin,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(shard.device),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(shard.device),
        server_peak_rss_kib=process_peak_rss_kib(),
    )


def serve(model_dir, host="127.0.0.1", port=9876, split=4, quantize=True, idle_timeout=180):
    import torch

    from .checkpoint import Checkpoint
    from .model import OutputHead, SplitShard

    if not math.isfinite(idle_timeout) or idle_timeout <= 0:
        raise ValueError("Idle timeout must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if not 0 <= split < 64:
        raise ValueError("split must be in 0..63")
    shard = SplitShard(model_dir, split, 64, "cuda:0", quantize=quantize)
    head = OutputHead(Checkpoint(model_dir), shard.config, shard.device) if split == 0 else None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"READY {host}:{server.getsockname()[1]}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(idle_timeout)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reset = False
                try:
                    while True:
                        msg, tensor, _ = recv_message(conn)
                        if msg.get("op") == "reset" and tensor is None:
                            if msg.get("split") != split:
                                raise ValueError("Client/server layer split mismatch")
                            shard.reset()
                            torch.cuda.reset_peak_memory_stats(shard.device)
                            reset = True
                            send_message(conn, {"op": "ok"})
                        elif msg.get("op") == "forward" and reset:
                            x = unpack_hidden(tensor)
                            position = msg.get("position")
                            if (x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != shard.hidden_size
                                    or type(position) is not int):
                                raise ValueError("Invalid hidden tensor or position")
                            out, metrics = gpu_forward(shard, x, position, head)
                            if head is None:
                                send_message(conn, {"op": "output", **metrics}, pack_hidden(out))
                            else:
                                send_message(conn, {"op": "token", "token_id": out, **metrics})
                        else:
                            raise ValueError("Unexpected operation or missing reset")
                except OSError:
                    pass
                except Exception as exc:
                    print(f"Request failed: {type(exc).__name__}: {exc}", flush=True)
                    try:
                        send_message(conn, {"op": "error", "message": str(exc)})
                    except OSError:
                        pass
                finally:
                    shard.reset()
