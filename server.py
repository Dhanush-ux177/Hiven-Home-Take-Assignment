#!/usr/bin/env python3
"""
server.py
100% Pure Python HTTP Server for @AppleSupport AI Agent & Benchmark Dashboard.
Requires zero external pip packages (uses Python standard library).

Serves:
- REST API endpoints for live agent inference and evaluation
- Golden evaluation set and benchmark metrics
- Frontend web dashboard from dist/ (SPA fallback)
"""

import os
import sys
import json
import mimetypes
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Add scripts directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from agent import AppleSupportAgent
from baselines import BaselineTrivial, BaselineSimple
from evaluator import run_full_benchmark

class AppleSupportHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200, "text/plain")
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self._set_headers(200, "text/html")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. API Endpoints
        if path == "/api/dataset":
            file_path = os.path.join(BASE_DIR, "data", "golden_eval_set.json")
            if os.path.exists(file_path):
                self._set_headers(200, "application/json")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Dataset not found"}).encode("utf-8"))
            return

        if path == "/api/benchmark":
            file_path = os.path.join(BASE_DIR, "data", "evaluation_results.json")
            if os.path.exists(file_path):
                self._set_headers(200, "application/json")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Benchmark not found"}).encode("utf-8"))
            return

        if path == "/api/exemplars":
            file_path = os.path.join(BASE_DIR, "data", "historical_exemplars.json")
            if os.path.exists(file_path):
                self._set_headers(200, "application/json")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Exemplars not found"}).encode("utf-8"))
            return

        if path == "/api/report":
            file_path = os.path.join(BASE_DIR, "REPORT.md")
            if os.path.exists(file_path):
                self._set_headers(200, "text/markdown; charset=utf-8")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self._set_headers(404, "text/plain")
            self.wfile.write(b"REPORT.md not found")
            return

        if path == "/api/health":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"status": "ok", "runtime": "python3"}).encode("utf-8"))
            return

        # 2. Static File Serving (from dist/)
        dist_dir = os.path.join(BASE_DIR, "dist")
        if not os.path.exists(dist_dir):
            # Fallback inline status page if dist has not been built yet
            self._set_headers(200, "text/html")
            html = """<!DOCTYPE html><html><head><title>@AppleSupport Agent</title></head>
            <body style="font-family:sans-serif;padding:40px;background:#090d16;color:#f8fafc">
            <h1>@AppleSupport Grounded AI Agent (Python Server)</h1>
            <p>Server running on Python 3. Build dist using <code>npm run build</code> for React UI.</p>
            <p>API endpoints active: <code>/api/benchmark</code>, <code>/api/dataset</code>, <code>/api/query</code></p>
            </body></html>"""
            self.wfile.write(html.encode("utf-8"))
            return

        # Map path to dist
        rel_path = path.lstrip("/")
        if not rel_path:
            rel_path = "index.html"
        full_path = os.path.join(dist_dir, rel_path)

        if os.path.isfile(full_path):
            ctype, _ = mimetypes.guess_type(full_path)
            if not ctype:
                ctype = "application/octet-stream"
            if full_path.endswith(".js"):
                ctype = "text/javascript"
            elif full_path.endswith(".css"):
                ctype = "text/css"
            self._set_headers(200, ctype)
            with open(full_path, "rb") as f:
                self.wfile.write(f.read())
            return

        # SPA fallback: serve dist/index.html
        index_path = os.path.join(dist_dir, "index.html")
        if os.path.isfile(index_path):
            self._set_headers(200, "text/html")
            with open(index_path, "rb") as f:
                self.wfile.write(f.read())
            return

        self._set_headers(404, "text/plain")
        self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        if path == "/api/query":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
                query = data.get("query", "").strip()
                use_llm = bool(data.get("llm", False))

                if not query:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "query parameter is required"}).encode("utf-8"))
                    return

                agent = AppleSupportAgent(use_llm=use_llm)
                b_trivial = BaselineTrivial()
                b_simple = BaselineSimple()

                res_agent = agent.process(query, force_llm=use_llm)
                res_t = b_trivial.process(query)
                res_s = b_simple.process(query)

                out = {
                    "customer_text": query,
                    "proposed_agent": res_agent,
                    "baseline_trivial": res_t,
                    "baseline_simple": res_s
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(out).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/run-eval":
            try:
                # Run evaluation in Python directly
                res = run_full_benchmark(save_path=os.path.join(BASE_DIR, "data", "evaluation_results.json"))
                self._set_headers(200)
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

def run_server(host="0.0.0.0", port=3000):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, AppleSupportHTTPHandler)
    print(f"🚀 AppleSupport Python Server running on http://{host}:{port}")
    print(f"   API endpoints ready. Serving static files from ./dist")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Python server...")
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="@AppleSupport AI Agent Python Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=3000, help="Port number to bind to (default: 3000)")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized arguments: {unknown}")
    run_server(host=args.host, port=args.port)
