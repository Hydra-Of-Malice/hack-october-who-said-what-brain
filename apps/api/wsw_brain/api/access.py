"""Who may change what. Reading, searching and asking stay open to anyone who can reach the server; changes
are limited:

- WSW_ADMIN_TOKEN (optional). When it is set, creating, uploading, ending, deleting and editing meetings,
  speakers and action items, and the live audio socket, need the operator's token: `Authorization: Bearer
  <token>`, an `X-WSW-Token` header, or the `wsw_token` cookie that POST /api/auth sets. Browsers send that
  cookie on every same-origin request and on the audio socket, so the web UI works unchanged once the
  operator has signed in (for example from /api/docs).
- Demo meetings are seeded by the server: only the operator may end or delete them or rename their
  speakers (without a token configured, nobody can through the API).
- New meetings are rate-limited per client address: WSW_CREATE_LIMIT per CREATE_WINDOW_S (0 = no limit).
- The audio socket refuses pages from other sites. CORS does not cover WebSockets.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.requests import HTTPConnection

from ..brand import REPO_ROOT

TOKEN_COOKIE = "wsw_token"
TOKEN_HEADER = "x-wsw-token"
TOKEN_COOKIE_MAX_AGE_S = 30 * 24 * 3600
CREATE_WINDOW_S = 600.0
MAX_TRACKED_CLIENTS = 4096
OPERATOR_ONLY = (
    "this server takes changes only from its operator: send the WSW_ADMIN_TOKEN as 'Authorization: Bearer "
    "<token>' or 'X-WSW-Token', or sign this browser in with POST /api/auth"
)
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class AccessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    admin_token: str = Field("", alias="WSW_ADMIN_TOKEN")
    create_limit: int = Field(30, alias="WSW_CREATE_LIMIT")
    allowed_hosts: str = Field("", alias="WSW_ALLOWED_HOSTS")  # comma-separated; empty = any Host header


class Access:
    def __init__(self, cfg: AccessSettings, cors_origins: list[str]) -> None:
        self.token = cfg.admin_token.strip() or None
        self.create_limit = max(0, cfg.create_limit)
        self.cors_origins = [o.rstrip("/") for o in cors_origins]
        self._created: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def required(self) -> bool:
        return self.token is not None

    def token_matches(self, value: str) -> bool:
        return self.token is not None and secrets.compare_digest(value.strip().encode(), self.token.encode())

    def is_operator(self, conn: HTTPConnection) -> bool:
        return any(self.token_matches(value) for value in _presented_tokens(conn))

    def check_operator(self, conn: HTTPConnection) -> None:
        if self.token is not None and not self.is_operator(conn):
            raise HTTPException(status_code=401, detail=OPERATOR_ONLY)

    def check_meeting_change(self, conn: HTTPConnection, meeting: dict[str, Any] | None) -> None:
        """Ending, deleting or renaming in a demo meeting is the operator's alone."""
        if meeting is not None and meeting.get("source") == "demo" and not self.is_operator(conn):
            raise HTTPException(
                status_code=403,
                detail="demo meetings are read-only; start a live meeting or upload a recording to try changes",
            )

    def check_create(self, conn: HTTPConnection) -> None:
        """Count one new meeting for the client's address (429 past the limit; the operator is exempt)."""
        if not self.create_limit or self.is_operator(conn):
            return
        client = conn.client.host if conn.client else "unknown"
        now = time.monotonic()
        with self._lock:
            if len(self._created) > MAX_TRACKED_CLIENTS:
                self._created = {
                    k: v for k, v in self._created.items() if v and now - v[-1] < CREATE_WINDOW_S
                }
            stamps = self._created.setdefault(client, deque())
            while stamps and now - stamps[0] >= CREATE_WINDOW_S:
                stamps.popleft()
            if len(stamps) >= self.create_limit:
                wait_s = int(CREATE_WINDOW_S - (now - stamps[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"too many new meetings from this address; try again in {wait_s} s",
                    headers={"Retry-After": str(wait_s)},
                )
            stamps.append(now)

    def origin_allowed(self, conn: HTTPConnection) -> bool:
        """May the page that opened this socket use it? Browsers always send Origin on a WebSocket."""
        origin = (conn.headers.get("origin") or "").rstrip("/")
        if not origin:
            return True  # not a browser (scripts/stream_wav.py)
        if "*" in self.cors_origins or origin in self.cors_origins:
            return True
        parts = urlsplit(origin)
        hosts = {
            conn.headers.get("host", "").strip().lower(),
            conn.headers.get("x-forwarded-host", "").split(",")[0].strip().lower(),
        }
        if parts.netloc and parts.netloc.lower() in hosts:
            return True  # the same site, directly or through a proxy that rewrote Host
        return (parts.hostname or "") in _LOOPBACK  # the Vite dev server proxies the socket from localhost


def _presented_tokens(conn: HTTPConnection) -> list[str]:
    values = [conn.headers.get(TOKEN_HEADER, ""), conn.cookies.get(TOKEN_COOKIE, "")]
    scheme, _, credentials = conn.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        values.append(credentials)
    return [v for v in values if v.strip()]
