"""Regenerate the two-speaker listener fixture (Windows only: needs the SAPI David and Zira voices).

    cd apps/api && uv run python tests/fixtures/build_fixture.py

Writes two_speakers.wav (16 kHz mono PCM16) and two_speakers.truth.json (turn order, text and times)
next to this file. Each turn is rendered by scripts/make_test_audio.ps1, trimmed of SAPI's leading and
trailing silence, and joined with GAP_S of low-level noise so the "silence" is not digital zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
GAP_S = 0.6
LEAD_S = 0.5
NOISE_STD = 0.0008
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
PS1 = REPO_ROOT / "scripts" / "make_test_audio.ps1"

DAVID = "Microsoft David Desktop"
ZIRA = "Microsoft Zira Desktop"

TURNS: list[tuple[str, str]] = [
    (DAVID, "Good morning everyone. Let's start with the pricing update for the new analytics dashboard."),
    (ZIRA, "Sure. The customer survey shows that most teams want a monthly plan instead of a yearly one."),
    (
        DAVID,
        "That makes sense. Then we should launch the monthly plan first and keep the yearly plan for later.",
    ),
    (ZIRA, "Agreed. I will update the pricing page and send the draft to marketing by Friday."),
    (DAVID, "Great. What about the mobile app? Is the login bug fixed yet?"),
    (ZIRA, "Not yet. Ravi is working on it, and he expects a fix early next week."),
    (DAVID, "Okay. Then we decide that the launch date moves to the twentieth of November."),
    (ZIRA, "Sounds good. I will tell the support team about the new launch date today."),
    (DAVID, "Perfect, thanks everyone."),
]


def _trim(samples: np.ndarray, floor: float = 0.004, pad_s: float = 0.04) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > floor)
    if loud.size == 0:
        raise ValueError("rendered turn is silent")
    pad = int(pad_s * SAMPLE_RATE)
    return samples[max(0, loud[0] - pad) : loud[-1] + pad]


def _render(workdir: Path) -> list[Path]:
    spec = [
        {"voice": voice, "text": text, "file": f"turn_{i:02d}.wav"} for i, (voice, text) in enumerate(TURNS)
    ]
    spec_path = workdir / "turns.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PS1),
        "-ScriptJson",
        str(spec_path),
        "-OutDir",
        str(workdir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"make_test_audio.ps1 failed:\n{result.stdout}\n{result.stderr}")
    return [workdir / item["file"] for item in spec]


def build(out_dir: Path = HERE, name: str = "two_speakers") -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    workdir = Path(tempfile.mkdtemp(prefix="wsw_fixture_"))
    try:
        files = _render(workdir)
        pieces: list[np.ndarray] = [np.zeros(int(LEAD_S * SAMPLE_RATE), dtype=np.float32)]
        cursor = LEAD_S
        truth = []
        for (voice, text), path in zip(TURNS, files, strict=True):
            samples, sr = sf.read(path, dtype="float32")
            if sr != SAMPLE_RATE or samples.ndim != 1:
                raise ValueError(f"{path.name}: expected 16 kHz mono, got {sr} Hz, shape {samples.shape}")
            turn = _trim(samples)
            speaker = "David" if voice == DAVID else "Zira"
            truth.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "t_start": round(cursor, 3),
                    "t_end": round(cursor + len(turn) / SAMPLE_RATE, 3),
                }
            )
            pieces += [turn, np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)]
            cursor += len(turn) / SAMPLE_RATE + GAP_S
        audio = np.concatenate(pieces)
        audio = audio + rng.normal(0.0, NOISE_STD, audio.shape).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        wav_path = out_dir / f"{name}.wav"
        truth_path = out_dir / f"{name}.truth.json"
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        truth_path.write_text(
            json.dumps(
                {
                    "sample_rate": SAMPLE_RATE,
                    "duration_s": round(len(audio) / SAMPLE_RATE, 3),
                    "turns": truth,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return wav_path, truth_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("The SAPI voices are Windows-only; the committed fixture is used on other platforms.")
    wav, truth = build()
    print(f"wrote {wav} ({wav.stat().st_size / 1e6:.2f} MB) and {truth}")
