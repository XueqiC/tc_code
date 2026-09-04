"""Local ollama.com proxy with API-key failover.

ollama.com enforces per-key session/weekly limits; on 2026-09-02 the tau2 user
simulator stalled for hours every time the key in use hit 429 while the other
key was fine. This proxy sits at http://localhost:<port>/v1/... and forwards to
https://ollama.com/v1/..., trying each configured key in turn on HTTP 429
(honouring Retry-After up to a short cap) so a lane never has to be restarted
just to switch keys. Keys are read from ~/.ollama_api_key* files, never logged.

Usage:
    python tools/ollama_proxy.py --port 8999 &
    OLLAMA_BASE_URL=http://localhost:8999 OLLAMA_API_KEY=proxy <lane command>
"""
from __future__ import annotations

import argparse
import glob
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request

UPSTREAM = "https://ollama.com"


def load_keys() -> list[str]:
    keys = []
    for path in sorted(glob.glob(os.path.expanduser("~/.ollama_api_key*"))):
        try:
            value = open(path).read().strip()
        except OSError:
            continue
        if value:
            keys.append(value)
    if not keys:
        sys.exit("no ~/.ollama_api_key* files found")
    return keys


class State:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.lock = threading.Lock()
        self.cooldown_until = [0.0] * len(keys)  # per-key 429 cooldown
        self.stats = {"ok": 0, "429": 0, "failover": 0, "errors": 0}

    def order(self) -> list[int]:
        now = time.time()
        ready = [i for i in range(len(self.keys)) if self.cooldown_until[i] <= now]
        cooling = [i for i in range(len(self.keys)) if self.cooldown_until[i] > now]
        return ready + sorted(cooling, key=lambda i: self.cooldown_until[i])


STATE: State


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; stats are printed periodically
        return

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        last_status, last_body, last_headers = 502, b'{"error":"proxy: no key succeeded"}', {}
        for attempt, idx in enumerate(STATE.order()):
            key = STATE.keys[idx]
            req = urllib.request.Request(UPSTREAM + self.path, data=body, method=method)
            for name in ("Content-Type", "Accept"):
                if self.headers.get(name):
                    req.add_header(name, self.headers[name])
            req.add_header("Authorization", f"Bearer {key}")
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = resp.read()
                    with STATE.lock:
                        STATE.stats["ok"] += 1
                        if attempt:
                            STATE.stats["failover"] += 1
                    self._reply(resp.status, data, resp.headers.get("Content-Type", "application/json"))
                    return
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                last_body = exc.read()
                last_headers = dict(exc.headers)
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        cool = min(float(retry_after), 900.0) if retry_after else 300.0
                    except ValueError:
                        cool = 300.0
                    with STATE.lock:
                        STATE.stats["429"] += 1
                        STATE.cooldown_until[idx] = time.time() + cool
                    continue  # try the next key
                with STATE.lock:
                    STATE.stats["errors"] += 1
                break
            except Exception as exc:  # network failure: try next key, else 502
                last_status, last_body = 502, json.dumps({"error": f"proxy: {exc}"}).encode()
                continue
        self._reply(last_status, last_body, last_headers.get("Content-Type", "application/json"))

    def _reply(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        if self.path == "/proxy/stats":
            with STATE.lock:
                payload = json.dumps({**STATE.stats, "cooldown": [max(0, int(t - time.time())) for t in STATE.cooldown_until]})
            self._reply(200, payload.encode(), "application/json")
            return
        self._forward("GET")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8999)
    args = parser.parse_args()
    global STATE
    STATE = State(load_keys())
    print(f"[ollama-proxy] {len(STATE.keys)} keys, listening on 127.0.0.1:{args.port}", flush=True)

    def report():
        while True:
            time.sleep(600)
            with STATE.lock:
                print(f"[ollama-proxy] stats {STATE.stats} cooldown={[max(0, int(t - time.time())) for t in STATE.cooldown_until]}", flush=True)

    threading.Thread(target=report, daemon=True).start()
    Server(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
