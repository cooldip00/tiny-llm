"""HTTP helpers shared by the model server and the replica router (D-031).

No torch here, on purpose: the router runs without it (tests/test_router.py checks that).
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import uuid
from pathlib import Path

from fastapi.responses import JSONResponse
from starlette.requests import Request

from tinyllm.obs import hash_client

# The router sends the visitor's hash (obs.hash_client: 16 hex characters) to a private replica.
CLIENT_HEADER = "x-tinyllm-client"
_CLIENT_HASH = re.compile(r"[0-9a-f]{16}")


def router_client(value: str | None) -> str | None:
    """The visitor hash in a router header, or None if it is missing or malformed."""
    return value if value and _CLIENT_HASH.fullmatch(value) else None


class RateLimiter:
    """Token bucket per client: `per_min` requests per minute, bursts up to `burst`.
    Called from the event loop only. per_min <= 0 disables it."""

    def __init__(self, per_min: float, burst: int):
        self.rate = per_min / 60.0
        self.burst = max(1, burst)
        self._b: dict[str, tuple[float, float]] = {}

    def check(self, client: str, now: float) -> float:
        """0 if allowed (and counted), else seconds until the next request is allowed."""
        if self.rate <= 0:
            return 0.0
        tokens, last = self._b.get(client, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            self._b[client] = (tokens - 1.0, now)
            if len(self._b) > 10_000:                     # forget clients whose bucket is full again
                self._b = {k: v for k, v in self._b.items() if v[0] + (now - v[1]) * self.rate < self.burst}
            return 0.0
        self._b[client] = (tokens, now)
        return (1.0 - tokens) / self.rate


_RID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class HeadersMiddleware:
    """Pure ASGI (keeps streaming and disconnects intact): every response gets X-Request-ID
    (the client's, if it sent a sane one) and X-Model-Version."""

    def __init__(self, app, default_version: str):
        self.app = app
        self.default_version = default_version

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex[:16]
        state = scope.setdefault("state", {})
        state["request_id"] = rid

        async def send_with_headers(msg):
            if msg["type"] == "http.response.start":
                version = state.get("model_version", self.default_version)
                msg = {**msg, "headers": [*msg.get("headers", []), (b"x-request-id", rid.encode()),
                                          (b"x-model-version", version.encode())]}
            await send(msg)

        await self.app(scope, receive, send_with_headers)


WEB = Path(__file__).resolve().parent / "web"


def read_page(name: str) -> str:
    """The demo and status pages (6.5b): static HTML + JS in src/tinyllm/web, read per request
    (small) so an edit shows on reload. They fetch the JSON endpoints with relative URLs."""
    return (WEB / name).read_text(encoding="utf-8")


def forwarded_ip(xff: str, hops: int = 1) -> str | None:
    """The client's address in an X-Forwarded-For header behind `hops` trusted proxies.

    Each proxy appends the address the request came from, so only the last `hops` entries
    are trustworthy; anything left of them is whatever the client sent (spoofable, so never
    the leftmost). With fewer entries than hops, all came from proxies: take the leftmost.
    """
    parts = [p.strip() for p in xff.split(",") if p.strip()]
    if not parts:
        return None
    return parts[-hops] if len(parts) >= hops else parts[0]


def client_hash(request: Request, trust_proxy: bool, hops: int, salt: str) -> str:
    """The salted visitor id for a request: from X-Forwarded-For behind a trusted proxy, else the peer."""
    ip = request.client.host if request.client else "unknown"
    if trust_proxy and (xff := request.headers.get("x-forwarded-for")):
        ip = forwarded_ip(xff, hops) or ip
    return hash_client(ip, salt)


def resolve_salt(salt: str | None = None) -> tuple[str, str]:
    """(salt, where it came from) for the client hash. The source is logged, never the salt.

    Order: an explicit salt; $TINYLLM_LOG_SALT (a Space secret, the same value in every
    replica Space, D-031); a hash of $SPACE_SIGNING_SECRET (HF's per-Space secret, stable
    across restarts, but different in each Space); else random per start.
    """
    if salt:
        return salt, "config"
    if s := os.environ.get("TINYLLM_LOG_SALT"):
        return s, "TINYLLM_LOG_SALT"
    if s := os.environ.get("SPACE_SIGNING_SECRET"):
        return hashlib.sha256(f"tinyllm-log-salt:{s}".encode()).hexdigest(), "SPACE_SIGNING_SECRET"
    return secrets.token_hex(16), "random"


def error_response(status: int, message: str, err_type: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": err_type, "code": status}},
                        status_code=status, headers=headers)
