"""Application frame metadata and Linux TCP socket counters."""
import socket
import struct
import time

# Stable offsets in Linux UAPI struct tcp_info. Short kernel results omit fields.
TCP_INFO_FIELDS = {
    "send_mss_bytes": (16, "I"),
    "smoothed_rtt_us": (68, "I"),
    "total_retransmissions": (100, "I"),
    "bytes_acked": (120, "Q"),
    "bytes_received": (128, "Q"),
    "segments_out": (136, "I"),
    "segments_in": (140, "I"),
    "data_segments_in": (152, "I"),
    "data_segments_out": (156, "I"),
}
COUNTERS = {"total_retransmissions", "bytes_acked", "bytes_received",
            "segments_out", "segments_in", "data_segments_in", "data_segments_out"}


def snapshot(sock):
    if not hasattr(socket, "TCP_INFO"):
        return {}
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 160)
    except OSError:
        return {}
    return {
        name: struct.unpack_from("=" + fmt, raw, offset)[0]
        for name, (offset, fmt) in TCP_INFO_FIELDS.items()
        if offset + struct.calcsize("=" + fmt) <= len(raw)
    }


def log_frame(emit, sock, direction, operation, size, stage, position=None,
              tensor=None, previous=None):
    tcp = snapshot(sock)
    record = dict(timestamp_unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                  direction=direction, operation=operation, phase=stage, position=position,
                  frame_bytes=size, tensor_bytes=tensor.nbytes if tensor is not None else 0,
                  tcp_info=tcp)
    if tensor is not None:
        record.update(tensor_shape=list(tensor.shape), wire_dtype=str(tensor.dtype))
    if previous is not None:
        record["tcp_exchange_delta"] = {
            name: tcp[name] - previous[name] for name in sorted(COUNTERS & tcp.keys() & previous.keys())
        }
    emit(record)
