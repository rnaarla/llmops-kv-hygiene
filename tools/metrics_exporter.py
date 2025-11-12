#!/usr/bin/env python3
"""
Simple HTTP exporter for Prometheus textfile metrics.
Serves the contents of a metrics file (default: forensics/metrics.prom) at /metrics.

Usage:
  METRICS_FILE=forensics/metrics.prom METRICS_PORT=8000 python tools/metrics_exporter.py
"""

import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

METRICS_FILE = os.environ.get("METRICS_FILE", "forensics/metrics.prom")
PORT = int(os.environ.get("METRICS_PORT", "8000"))
BIND = os.environ.get("METRICS_BIND", "0.0.0.0")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # quiet
        try:
            msg = format % args
        except Exception:  # pragma: no cover - defensive formatting fallback
            # We intentionally avoid raising here; fall back to raw format string
            msg = format
        sys.stderr.write(f"{self.address_string()} - - [{self.log_date_time_string()}] {msg}\n")

    def do_GET(self):  # noqa: N802 - required method name by BaseHTTPRequestHandler
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        # Read METRICS_FILE from environment at request time to support runtime configuration
        metrics_file = os.environ.get("METRICS_FILE", METRICS_FILE)
        try:
            with open(metrics_file, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            data = b""  # empty until produced
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":  # pragma: no cover - integration path
    httpd = HTTPServer((BIND, PORT), Handler)
    print(f"serving metrics from {METRICS_FILE} on {BIND}:{PORT} at /metrics", flush=True)
    httpd.serve_forever()
