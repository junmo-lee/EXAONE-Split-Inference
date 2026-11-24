"""Length-delimited, tensor-only TCP protocol (not a secure transport)."""
import math
import json
import socket
import struct
import time
from typing import Any

import numpy as np

MAX_FRAME = 128 * 1024 * 1024
DTYPES = {"float16": np.dtype("<f2"), "float32": np.dtype("<f4"), "uint16": np.dtype("<u2")}


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = sock.recv(min(1 << 20, count - len(chunks)))
        if not chunk:
            raise ConnectionError("Connection closed during frame")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, payload: dict[str, Any], tensor=None) -> int:
    header = dict(payload)
    raw = b""
    if tensor is not None:
        arr = np.asarray(tensor)
        if arr.dtype.name not in DTYPES:
            raise TypeError(f"Unsupported dtype: {arr.dtype}")
        arr = np.ascontiguousarray(arr, dtype=DTYPES[arr.dtype.name])
        header.update(shape=list(arr.shape), dtype=str(arr.dtype))
        raw = arr.tobytes()
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) + len(raw) > MAX_FRAME or len(encoded) > 65536:
        raise ValueError("Frame too large")
    prefix = struct.pack("!IQ", len(encoded), len(raw))
    sock.sendall(prefix + encoded + raw)
    return len(prefix) + len(encoded) + len(raw)


def recv_message(sock: socket.socket):
    header_size, raw_size = struct.unpack("!IQ", recv_exact(sock, 12))
    if header_size > 65536 or header_size + raw_size > MAX_FRAME:
        raise ValueError("Invalid frame length")
    header = json.loads(recv_exact(sock, header_size))
    if not isinstance(header, dict):
        raise ValueError("Frame header must be a JSON object")
    if not raw_size:
        return header, None, 12 + header_size
    dtype = DTYPES.get(header.get("dtype"))
    shape = header.get("shape")
    if dtype is None or not isinstance(shape, list) or not 1 <= len(shape) <= 8 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Invalid tensor metadata")
    if math.prod(shape) * dtype.itemsize != raw_size:
        raise ValueError("Tensor length mismatch")
    raw = recv_exact(sock, raw_size)
    return header, np.frombuffer(raw, dtype=dtype).reshape(shape).copy(), 12 + header_size + raw_size


class SimulatedLink:
    """Adds an *aggregate* round-trip delay once per request/reply pair."""

    def __init__(self, rtt_ms: float, bandwidth_mbps: float = 0):
        if any(not math.isfinite(x) or x < 0 for x in (rtt_ms, bandwidth_mbps)):
            raise ValueError("RTT and bandwidth must be finite and nonnegative")
        self.rtt_ms = rtt_ms
        self.bandwidth_mbps = bandwidth_mbps

    def delay(self, outbound_bytes: int, inbound_bytes: int):
        seconds = self.rtt_ms / 1000.0
        if self.bandwidth_mbps:
            seconds += 8 * (outbound_bytes + inbound_bytes) / (self.bandwidth_mbps * 1e6)
        begin = time.perf_counter()
        if seconds:
            time.sleep(seconds)
        return time.perf_counter() - begin if seconds else 0.0


def pack_hidden(tensor):
    """Transport BF16 bit patterns without a lossy FP16 conversion."""
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()


def unpack_hidden(array):
    import torch

    if array is None or array.dtype != DTYPES["uint16"]:
        raise ValueError("Expected BF16 hidden state encoded as uint16")
    return torch.from_numpy(array).view(torch.bfloat16)
