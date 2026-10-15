"""FastAPI wiring: REST, Server-Sent Events and the live audio WebSocket."""

from .app import create_app

__all__ = ["create_app"]
