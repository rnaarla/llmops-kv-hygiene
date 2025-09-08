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
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format%args))

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        try:
            with open(METRICS_FILE, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            data = b""  # empty until produced
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

if __name__ == "__main__":
    httpd = HTTPServer((BIND, PORT), Handler)
    print(f"serving metrics from {METRICS_FILE} on {BIND}:{PORT} at /metrics", flush=True)
    httpd.serve_forever()
