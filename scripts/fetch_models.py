"""Download the small ONNX models the listener needs into ./models (git-ignored).

Run from the repo root:  uv run --project apps/api python scripts/fetch_models.py

Large models are NOT fetched here:
- Gemma 4 runs in Ollama (`ollama pull gemma4:e4b`) or via the Gemini API.
- faster-whisper and EmbeddingGemma download into the Hugging Face cache on first use.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# Official k2-fsa/sherpa-onnx release assets (the "recongition" typo is the real tag name).
FILES = {
    "silero_vad.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "speaker_embedding.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    ),
    "speaker_embedding_en.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "nemo_en_titanet_small.onnx"
    ),
}


def fetch(name: str, url: str) -> None:
    dest = MODELS / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"ok   {name} ({dest.stat().st_size / 1e6:.1f} MB, cached)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get  {name} <- {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"ok   {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        fetch(name, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
