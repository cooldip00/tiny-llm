"""Smoke-check a running tinyllm server: the Docker image in CI, a local container, or the Space (6.5b).

Run:  python scripts/smoke_server.py --url http://127.0.0.1:7860 [--wait 120] [--norm-backend torch]

Standard library only, so it runs anywhere. Waits for /ready (the model loads at start), then checks:
  1. GET /health is {"status": "ok"}, GET /ready is {"ready": true}
  2. one non-streamed completion: 200, finish_reason, usage.completion_tokens >= 1
  3. GET /stats: the request is counted; each variant reports the expected norm_backend
Exits non-zero with the first failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--wait", type=float, default=120.0, help="seconds to wait for /ready")
    p.add_argument("--norm-backend", default="torch", help="expected in /stats (CPU: torch)")
    p.add_argument("--prompt", default="Once upon a time")
    args = p.parse_args()
    base = args.url.rstrip("/")

    deadline, last = time.monotonic() + args.wait, None
    while True:
        try:
            status, body = call(f"{base}/ready", timeout=5)
            if status == 200 and body == {"ready": True}:
                break
            last = f"{status} {body}"
        except (OSError, ValueError) as e:
            last = repr(e)
        if time.monotonic() > deadline:
            check(False, f"/ready within {args.wait:.0f} s (last: {last})")
        time.sleep(2)
    check(True, "/ready")
    check(call(f"{base}/health") == (200, {"status": "ok"}), "/health")

    status, out = call(f"{base}/v1/completions", {"prompt": args.prompt, "max_tokens": 24, "temperature": 0})
    choice = (out or {}).get("choices", [{}])[0]
    check(status == 200 and choice.get("finish_reason") in ("stop", "length")
          and out["usage"]["completion_tokens"] >= 1,
          f"completion: {status}, {out and out.get('usage')}, text {choice.get('text', '')[:60]!r}")

    status, st = call(f"{base}/stats")
    reqs = st["windows"]["5m"]["requests"] if status == 200 else 0
    check(status == 200 and reqs >= 1, f"/stats counts the request ({reqs})")
    backends = {name: v.get("norm_backend") for name, v in st["variants"].items()}
    check(bool(backends) and set(backends.values()) == {args.norm_backend},
          f"/stats norm_backend {backends} (expected {args.norm_backend!r})")
    print("smoke check passed")


if __name__ == "__main__":
    main()
