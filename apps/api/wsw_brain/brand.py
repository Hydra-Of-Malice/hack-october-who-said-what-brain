"""The product name lives in ONE place: /brand.json at the repo root.

Rename the product by editing that file; the API and the web UI both read it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BRAND_FILE = REPO_ROOT / "brand.json"


@lru_cache(maxsize=1)
def brand() -> dict[str, str]:
    return json.loads(BRAND_FILE.read_text(encoding="utf-8"))


def product_name() -> str:
    return brand()["name"]
