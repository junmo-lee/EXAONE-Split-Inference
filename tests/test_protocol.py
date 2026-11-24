import socket
import threading
import unittest

import numpy as np

from exaone_split.protocol import SimulatedLink, recv_message, send_message


class TestProtocol(unittest.TestCase):
    def test_tensor_roundtrip(self):
        a, b = socket.socketpair()
        arr = np.arange(30, dtype=np.float16).reshape(1, 3, 10)
        try:
            thread = threading.Thread(target=lambda: send_message(a, {"op":"forward", "position":2}, arr))
            thread.start()
            meta, decoded, size = recv_message(b)
            thread.join()
            self.assertEqual(meta["position"], 2)
            self.assertEqual(size, 12 + len(arr.tobytes()) + len(__import__('json').dumps(meta, separators=(',', ':')).encode()))
            np.testing.assert_array_equal(decoded, arr)
        finally:
            a.close()
            b.close()

    def test_empty_response(self):
        a, b = socket.socketpair()
        try:
            send_message(a, {"op":"reset"})
            msg, arr, _ = recv_message(b)
            self.assertEqual(msg["op"], "reset")
            self.assertIsNone(arr)
        finally:
            a.close()
            b.close()

    def test_simulated_latency(self):
        import time
        before = time.perf_counter()
        SimulatedLink(1).delay(100, 100)
        self.assertGreaterEqual(time.perf_counter()-before, 0.0009)


if __name__ == "__main__":
    unittest.main()
