"""
neuroforge.server.app
=====================

A tiny stdlib HTTP server that:
  * serves the static Three.js frontend from  web/
  * serves the generated anatomy as JSON at   /api/scene
  * exposes the simulation engine at          /api/sim/*

Deliberately dependency-free (http.server, json) so the application runs on
any Python 3.10+ with nothing to install. The browser is used purely as a
render surface: all state, all anatomy and all simulation logic live here in
Python.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Any, Callable, Dict
from urllib.parse import urlparse

WEB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "web")


class ClientGone(Exception):
    """The browser closed the connection mid-stream. Not an error."""


class Api:
    """Route table. Handlers take a dict payload and return a JSON-able dict."""

    def __init__(self) -> None:
        self._get: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._post: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        # Streaming handlers get the raw request handler and write to it
        # themselves, because server-sent events must be flushed as they are
        # produced rather than buffered into one response body.
        self._stream: Dict[str, Callable[[Dict[str, Any], Any], None]] = {}

    def get(self, path: str):
        def deco(fn):
            self._get[path] = fn
            return fn
        return deco

    def post(self, path: str):
        def deco(fn):
            self._post[path] = fn
            return fn
        return deco

    def stream(self, path: str):
        def deco(fn):
            self._stream[path] = fn
            return fn
        return deco


API = Api()


class Handler(SimpleHTTPRequestHandler):
    api: Api = API

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB_ROOT, **kw)

    # ---- quieter logging --------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in (self.path or "") and "GET /api/sim/state" not in fmt % args:
            print(f"[api] {fmt % args}", flush=True)

    # ---- helpers ------------------------------------------------------
    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ---- server-sent events -------------------------------------------
    def sse_begin(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        # Chunked would need manual framing; Connection: close plus an
        # explicit end-of-stream event is simpler and works everywhere.
        self.end_headers()

    def sse(self, event: str, data: Any) -> None:
        """Send one event. Returns quietly if the client has gone away."""
        try:
            payload = json.dumps(data)
            self.wfile.write(f"event: {event}\ndata: {payload}\n\n"
                             .encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            raise ClientGone()

    # ---- routing ------------------------------------------------------
    def do_GET(self) -> None:            # noqa: N802
        path = urlparse(self.path).path
        fn = self.api._get.get(path)
        if fn:
            try:
                self._send_json(fn({}))
            except Exception as exc:                      # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(exc)}, 500)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:           # noqa: N802
        path = urlparse(self.path).path

        stream_fn = self.api._stream.get(path)
        if stream_fn:
            payload = self._read_json()
            try:
                stream_fn(payload, self)
            except ClientGone:
                pass                      # user closed the panel mid-reply
            except Exception as exc:      # noqa: BLE001
                import traceback
                traceback.print_exc()
                # Headers are likely already sent, so an HTTP error code is
                # not available. Report through the stream instead.
                try:
                    self.sse("error", {"message": str(exc)})
                except Exception:         # noqa: BLE001
                    pass
            return

        fn = self.api._post.get(path)
        if not fn:
            self._send_json({"error": "no such endpoint"}, 404)
            return
        try:
            self._send_json(fn(self._read_json()))
        except Exception as exc:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)

    def end_headers(self) -> None:
        # allows the page to use SharedArrayBuffer-free WebGL2 features and
        # stops aggressive caching of the dev assets
        if self.path.endswith((".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def serve(port: int = 8770, open_browser: bool = True,
          subdivisions: int = 6) -> None:
    # Import here so the route decorators have run.
    from . import routes                                   # noqa: F401
    from ..anatomy.build import get_scene

    print("=" * 66)
    print(" NeuroForge — interactive 3-D neuroscience simulation")
    print(" Educational visual metaphor. Not a brain scanner, not a")
    print(" diagnostic tool. Every value shown is a simulation variable.")
    print("=" * 66)

    routes.SCENE = get_scene(subdivisions)                 # warm the cache

    # Threaded: a single slow request (a language model generating a reply,
    # a large scene serialising) must not stall the whole application. The
    # simulation state this exposes to concurrent handlers is guarded by a
    # lock in routes.py -- see routes.STATE_LOCK.
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True    # don't let in-flight requests block Ctrl+C
    url = f"http://127.0.0.1:{port}/"
    print(f"\n[server] listening on {url}   (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down")
        httpd.server_close()
