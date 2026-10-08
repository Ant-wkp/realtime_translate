
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.http11 import Request, Response
from websockets.datastructures import Headers

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
        self._ready_evt = threading.Event()
        self._bind_error: BaseException | None = None

    def start(self, timeout: float = 5.0) -> None:
        self._thread.start()
        if not self._ready_evt.wait(timeout):
            raise RuntimeError("web server did not start in time")
        if self._bind_error is not None:
            raise RuntimeError(
                f"cannot bind {self.host}:{self.port}: {self._bind_error}"
            ) from self._bind_error

    def stop(self) -> None:
        self._stop_evt.set()

    def broadcast(self, msg: dict) -> None:
        self.out_queue.put(msg)

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

        async def http_handler(connection, request: Request):
            path = request.path
            if path in ("/", "/index.html"):
                body = (web_dir / "index.html").read_bytes()
                return Response(
                    200, "OK",
                    headers=Headers([("Content-Type", "text/html; charset=utf-8"),
                                     ("Content-Length", str(len(body)))]),
                    body=body,
                )
            return None

        try:
            async with serve(
                handler,
                self.host,
                self.port,
                process_request=http_handler,
                ping_interval=20,
            ):
                self._ready_evt.set()
                pump = asyncio.create_task(self._pump())
                try:
                    await asyncio.Future()
                finally:
                    pump.cancel()
        except OSError as exc:
            self._bind_error = exc
            self._ready_evt.set()

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
