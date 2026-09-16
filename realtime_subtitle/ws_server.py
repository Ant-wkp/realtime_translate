"""Broadcast server: one asyncio thread, thread-safe outbound queue.

Producers (main loop, ASR/translation workers) call broadcast() which is
always a cheap queue put. A _pump coroutine drains the queue at 10 ms
intervals and pushes JSON to every /ws client. Inbound commands from the
frontend land in a cmd_queue consumed by the main loop.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path

import websockets
from websockets.server import serve


class BroadcastServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 cmd_queue: queue.Queue | None = None):
        self.host = host
        self.port = port
        self.clients = set()
        self.out_queue: queue.Queue = queue.Queue()
        self.cmd_queue = cmd_queue or queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="BroadcastServer")
        self._loop = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    # called from any thread - cheap ------------------------------------
    def broadcast(self, msg: dict) -> None:
        self.out_queue.put(msg)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        web_dir = Path(__file__).parent / "web"

        async def handler(ws):
            self.clients.add(ws)
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self.cmd_queue.put(msg)
            finally:
                self.clients.discard(ws)

        async def http_handler(path, headers):
            if path in ("/", "/index.html"):
                body = (web_dir / "index.html").read_bytes()
                return (websockets.http11.Response(
                    200, "OK", headers=[("Content-Type", "text/html; charset=utf-8")]),
                    body)
            return None  # fall back -> 404

        async with serve(
            handler,
            self.host,
            self.port,
            process_request=http_handler,
            ping_interval=20,
        ):
            pump = asyncio.create_task(self._pump())
            try:
                await asyncio.Future()  # run until stopped
            finally:
                pump.cancel()

    async def _pump(self) -> None:
        while True:
            try:
                msg = self.out_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if not self.clients:
                continue
            raw = json.dumps(msg, ensure_ascii=False)
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(raw)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)
