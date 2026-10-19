from __future__ import annotations

import json
import unittest

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from realtime_translate.ws_server import BroadcastServer


class TestWebServer(unittest.TestCase):
    def test_broadcast_command_and_stop_with_connected_client(self):
        server = BroadcastServer(host="127.0.0.1", port=0)
        server.start()
        port = server.port
        try:
            with connect(f"ws://127.0.0.1:{port}/ws", proxy=None, open_timeout=1, close_timeout=1) as ws:
                ws.send(json.dumps({"cmd": "get_zones"}))
                self.assertEqual(server.cmd_queue.get(timeout=1)["cmd"], "get_zones")
                server.broadcast({"type": "final", "id": "seg", "text": "你好"})
                self.assertEqual(json.loads(ws.recv(timeout=1))["text"], "你好")
                server.stop()
                server.stop()
                server.join(2)
                self.assertFalse(server.is_alive())
                with self.assertRaises(ConnectionClosed):
                    ws.recv(timeout=1)
        finally:
            server.stop()
            server.join(2)
        rebound = BroadcastServer(host="127.0.0.1", port=port)
        try:
            rebound.start()
            self.assertTrue(rebound.is_alive())
        finally:
            rebound.stop()
            rebound.join(2)
        self.assertFalse(rebound.is_alive())

    def test_bind_failure_leaves_no_thread(self):
        first = BroadcastServer(host="127.0.0.1", port=0)
        first.start()
        second = BroadcastServer(host="127.0.0.1", port=first.port)
        try:
            with self.assertRaisesRegex(RuntimeError, "cannot bind"):
                second.start()
        finally:
            second.stop()
            second.join(2)
            first.stop()
            first.join(2)
        self.assertFalse(second.is_alive())

    def test_stop_before_start_and_idle_stop(self):
        unused = BroadcastServer(host="127.0.0.1", port=0)
        unused.stop()
        unused.join(0)
        with self.assertRaisesRegex(RuntimeError, "already been stopped"):
            unused.start()
        server = BroadcastServer(host="127.0.0.1", port=0)
        try:
            server.start()
        finally:
            server.stop()
            server.join(2)
        self.assertFalse(server.is_alive())


if __name__ == "__main__":
    unittest.main()
