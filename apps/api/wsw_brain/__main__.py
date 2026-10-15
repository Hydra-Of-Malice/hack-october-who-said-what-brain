"""Run the API (and the built web UI): python -m wsw_brain [--host 127.0.0.1] [--port 8000] [--reload]"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .brand import product_name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m wsw_brain", description=f"{product_name()} API server")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (0.0.0.0 for every interface)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart when wsw_brain source files change")
    parser.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )
    args = parser.parse_args(argv)
    uvicorn.run(
        "wsw_brain.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parent)] if args.reload else None,
        log_level=args.log_level,
        ws_max_size=1024 * 1024,
    )


if __name__ == "__main__":
    main()
