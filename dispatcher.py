"""
Dispatcher server that forwards digit generation requests to backend servers
according to a mapping file.

Usage:
    python dispatcher.py --mapping digit_mapping.json [--port 8000]

Mapping file format (JSON):
    { "0": "http://host1:5000", "1": "http://host1:5000", ..., "5": "http://host2:5000", ... }

Request:
    GET /generate?digit=3

Response:
    PNG image proxied from the appropriate backend server.
"""

import argparse
import json
import sys
import requests
from flask import Flask, request, Response, jsonify


def load_mapping(path: str) -> dict[str, str]:
    with open(path) as f:
        mapping = json.load(f)
    missing = [str(d) for d in range(10) if str(d) not in mapping]
    if missing:
        print(f"Warning: no server mapped for digit(s): {', '.join(missing)}")
    return mapping


def create_app(mapping_path: str) -> Flask:
    mapping = load_mapping(mapping_path)
    print("Digit routing:")
    for digit, server in sorted(mapping.items()):
        print(f"  {digit} -> {server}")

    app = Flask(__name__)

    @app.get("/generate")
    def generate():
        digit_str = request.args.get("digit")
        if digit_str is None:
            return jsonify(error="'digit' query parameter is required"), 400
        if digit_str not in mapping:
            return jsonify(error=f"digit '{digit_str}' is not in the mapping"), 400

        backend = mapping[digit_str]
        try:
            resp = requests.get(f"{backend}/generate", params={"digit": digit_str}, timeout=60)
        except requests.ConnectionError:
            return jsonify(error=f"could not reach backend {backend}"), 502
        except requests.Timeout:
            return jsonify(error=f"backend {backend} timed out"), 504

        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/octet-stream"),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Digit generation dispatcher")
    parser.add_argument("--mapping", type=str, default="digit_mapping.json", help="Path to digit-to-server mapping JSON")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    app = create_app(args.mapping)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
