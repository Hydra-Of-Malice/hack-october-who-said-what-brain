"""Synthesise the demo meeting scripts into 16 kHz mono audio, one distinct Kokoro voice per speaker.

The output is fed through the REAL listener (VAD -> ASR -> speaker clustering -> graph writes), so the
demo proves diarization instead of replaying a pre-baked transcript. For each script it writes:

  demo/audio/<id>.ogg               Ogg Opus, 16 kHz mono (small enough to commit)
  demo/meetings/<id>.truth.json     ground-truth turns {speaker, t_start, t_end, text} for scoring

kokoro-onnx is not a project dependency. Run it in a throwaway uv overlay from the repo root:

  uv run --project apps/api --no-sync --with kokoro-onnx python demo/make_audio.py --download

`--download` fetches kokoro-v1.0.onnx (325 MB) and voices-v1.0.bin (28 MB) from the official
kokoro-onnx GitHub release into models/kokoro/ (git-ignored) if they are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
MEETINGS_DIR = DEMO_DIR / "meetings"
AUDIO_DIR = DEMO_DIR / "audio"
KOKORO_DIR = ROOT / "models" / "kokoro"
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICES = "voices-v1.0.bin"

SAMPLE_RATE = 16_000
LEAD_IN_S = 0.6
TAIL_S = 1.0
PAUSE_RANGE_S = (0.3, 0.9)
TURN_RANGE_S = (4.0, 15.0)  # the scripts aim for this; outside it we warn, not fail
TARGET_RMS_DBFS = -20.0
PEAK_CEILING = 0.89  # about -1 dBFS
TRIM_BELOW_PEAK_DB = 40.0
TRIM_FLOOR_DBFS = -60.0
TRIM_MARGIN_S = 0.02
FRAME_S = 0.01
OPUS_COMPRESSION = 0.88  # libsndfile maps 0..1 onto Opus bitrate (high..low); ~36 kb/s at 16 kHz mono


@dataclass(frozen=True)
class Voice:
    voice: str
    speed: float
    lang: str


@dataclass(frozen=True)
class TruthTurn:
    seq: int
    speaker: str
    t_start: float
    t_end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "speaker": self.speaker, "t_start": round(self.t_start, 3),
                "t_end": round(self.t_end, 3), "text": self.text}


# ---------------------------------------------------------------------------
# Script loading and validation
# ---------------------------------------------------------------------------


def lang_for(voice: str) -> str:
    """Kokoro voice ids start with a language letter: 'b' = British English; everything else is read
    as American English (the Hindi 'h' voices then speak English with an Indian accent)."""
    return "en-gb" if voice.startswith("b") else "en-us"


def load_script(path: Path) -> dict[str, Any]:
    try:
        script = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: cannot read meeting script ({exc})") from exc
    validate_script(script, path.name)
    return script


def validate_script(script: dict[str, Any], where: str) -> None:
    for key in ("id", "title", "speakers", "turns"):
        if not script.get(key):
            raise ValueError(f"{where}: missing or empty '{key}'")
    names = [s.get("name") for s in script["speakers"]]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"{where}: speaker names must be unique non-empty strings, got {names}")
    for spk in script["speakers"]:
        if not isinstance(spk.get("voice"), str) or not spk["voice"]:
            raise ValueError(f"{where}: speaker {spk['name']} has no 'voice'")
        speed = spk.get("speed", 1.0)
        if not 0.5 <= float(speed) <= 2.0:
            raise ValueError(f"{where}: speaker {spk['name']} speed {speed} is outside 0.5..2.0")
    lo, hi = PAUSE_RANGE_S
    for i, turn in enumerate(script["turns"]):
        if turn.get("speaker") not in names:
            raise ValueError(f"{where}: turn {i} speaker {turn.get('speaker')!r} is not in speakers {names}")
        if not str(turn.get("text", "")).strip():
            raise ValueError(f"{where}: turn {i} has empty text")
        pause = turn.get("pause_after_s")
        if not isinstance(pause, (int, float)) or not lo <= pause <= hi:
            raise ValueError(f"{where}: turn {i} pause_after_s={pause!r} must be a number in {lo}..{hi}")


def voices_for(script: dict[str, Any], overrides: dict[str, str]) -> dict[str, Voice]:
    out: dict[str, Voice] = {}
    for spk in script["speakers"]:
        voice = overrides.get(spk["name"], spk["voice"])
        out[spk["name"]] = Voice(voice=voice, speed=float(spk.get("speed", 1.0)), lang=spk.get("lang") or lang_for(voice))
    return out


def find_scripts(selected: list[str]) -> list[Path]:
    all_scripts = sorted(p for p in MEETINGS_DIR.glob("*.json") if not p.name.endswith(".truth.json"))
    if not selected:
        return all_scripts
    by_id = {p.stem: p for p in all_scripts}
    paths: list[Path] = []
    for item in selected:
        path = by_id.get(item) or Path(item)
        if not path.is_file():
            raise ValueError(f"unknown meeting {item!r}; choose from {sorted(by_id)} or pass a path")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Kokoro
# ---------------------------------------------------------------------------


def ensure_models(model_dir: Path, download: bool) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = (model_dir / KOKORO_MODEL, model_dir / KOKORO_VOICES)
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            continue
        if not download:
            raise FileNotFoundError(
                f"{path} is missing. Re-run with --download, or fetch {KOKORO_RELEASE}{path.name} into {model_dir}"
            )
        part = path.with_suffix(path.suffix + ".part")
        print(f"get  {path.name} <- {KOKORO_RELEASE}{path.name}", flush=True)
        urllib.request.urlretrieve(KOKORO_RELEASE + path.name, part)
        part.replace(path)
        print(f"ok   {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return paths


def load_kokoro(model: Path, voices: Path) -> Any:
    try:
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        raise RuntimeError(
            "kokoro-onnx is not installed. Run from the repo root with:\n"
            "  uv run --project apps/api --no-sync --with kokoro-onnx python demo/make_audio.py"
        ) from exc
    return Kokoro(str(model), str(voices))


def check_voices(kokoro: Any, voices: dict[str, Voice]) -> None:
    available = set(kokoro.get_voices())
    unknown = sorted({v.voice for v in voices.values()} - available)
    if unknown:
        raise ValueError(f"unknown Kokoro voice(s) {unknown}; available: {sorted(available)}")


def synthesise(kokoro: Any, text: str, voice: Voice) -> np.ndarray:
    samples, rate = kokoro.create(text, voice=voice.voice, speed=voice.speed, lang=voice.lang)
    audio = np.asarray(samples, dtype=np.float32)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Signal shaping
# ---------------------------------------------------------------------------


def frame_rms_db(audio: np.ndarray) -> np.ndarray:
    hop = int(FRAME_S * SAMPLE_RATE)
    n = len(audio) // hop
    if n == 0:
        return np.full(1, -120.0)
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Cut leading/trailing near-silence so truth timings mark where speech actually is."""
    db = frame_rms_db(audio)
    threshold = max(db.max() - TRIM_BELOW_PEAK_DB, TRIM_FLOOR_DBFS)
    active = np.flatnonzero(db > threshold)
    if active.size == 0:
        raise ValueError("synthesised turn is silent")
    hop = int(FRAME_S * SAMPLE_RATE)
    margin = int(TRIM_MARGIN_S * SAMPLE_RATE)
    start = max(0, active[0] * hop - margin)
    end = min(len(audio), (active[-1] + 1) * hop + margin)
    return audio[start:end]


def normalise(audio: np.ndarray) -> np.ndarray:
    """Match speech loudness across voices (RMS over voiced frames), then keep peaks below the ceiling."""
    db = frame_rms_db(audio)
    voiced = db[db > db.max() - 30.0]
    current = 10.0 * np.log10(np.mean(10.0 ** (voiced / 10.0)))
    out = audio * (10.0 ** ((TARGET_RMS_DBFS - current) / 20.0))
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_CEILING:
        out *= PEAK_CEILING / peak
    return out.astype(np.float32)


def room_tone(n: int, dbfs: float, seed: int) -> np.ndarray:
    """Quiet, slightly low-passed noise so pauses are not digital zero (closer to a real room)."""
    if dbfs <= -120.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    smooth = np.convolve(white, np.ones(4) / 4.0, mode="same")
    smooth *= (10.0 ** (dbfs / 20.0)) / max(float(np.sqrt(np.mean(smooth**2))), 1e-12)
    return smooth.astype(np.float32)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(kokoro: Any, script: dict[str, Any], voices: dict[str, Voice],
           room_db: float, seed: int) -> tuple[np.ndarray, list[TruthTurn]]:
    pieces: list[np.ndarray] = [np.zeros(int(LEAD_IN_S * SAMPLE_RATE), dtype=np.float32)]
    cursor = len(pieces[0])
    truth: list[TruthTurn] = []
    for seq, turn in enumerate(script["turns"]):
        try:
            speech = normalise(trim_silence(synthesise(kokoro, turn["text"], voices[turn["speaker"]])))
        except Exception as exc:  # kokoro/espeak raise assorted types; name the turn that failed
            raise RuntimeError(f"{script['id']} turn {seq} ({turn['speaker']}): synthesis failed: {exc}") from exc
        start = cursor
        cursor += len(speech)
        truth.append(TruthTurn(seq, turn["speaker"], start / SAMPLE_RATE, cursor / SAMPLE_RATE, turn["text"]))
        pause = np.zeros(round(float(turn["pause_after_s"]) * SAMPLE_RATE), dtype=np.float32)
        pieces += [speech, pause]
        cursor += len(pause)
        print(f"  {seq:3d} {turn['speaker']:<6s} {start / SAMPLE_RATE:7.2f}-{cursor / SAMPLE_RATE:7.2f}s "
              f"{turn['text'][:60]}", flush=True)
    pieces.append(np.zeros(int(TAIL_S * SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(pieces)
    audio += room_tone(len(audio), room_db, seed)
    return np.clip(audio, -1.0, 1.0), truth


def write_audio(path: Path, audio: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "opus":
        sf.write(path, audio, SAMPLE_RATE, format="OGG", subtype="OPUS", compression_level=OPUS_COMPRESSION)
    elif fmt == "vorbis":
        sf.write(path, audio, SAMPLE_RATE, format="OGG", subtype="VORBIS", compression_level=0.6)
    elif fmt == "wav":
        sf.write(path, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    else:
        raise ValueError(f"unknown format {fmt!r}")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def truth_document(script: dict[str, Any], voices: dict[str, Voice], turns: list[TruthTurn],
                   audio_path: Path, duration_s: float) -> dict[str, Any]:
    return {
        "meeting_id": script["id"],
        "title": script["title"],
        "audio": display_path(audio_path),
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration_s, 3),
        "engine": f"Kokoro-82M v1.0 ({KOKORO_MODEL}) via kokoro-onnx {package_version('kokoro-onnx')}",
        "voices": {name: {"voice": v.voice, "speed": v.speed, "lang": v.lang} for name, v in voices.items()},
        "turns": [t.to_dict() for t in turns],
    }


def summarise(script: dict[str, Any], turns: list[TruthTurn], audio_path: Path, duration_s: float) -> None:
    talk: dict[str, float] = {}
    for t in turns:
        talk[t.speaker] = talk.get(t.speaker, 0.0) + (t.t_end - t.t_start)
    lo, hi = TURN_RANGE_S
    outside = [f"#{t.seq} {t.speaker} {t.t_end - t.t_start:.1f}s" for t in turns if not lo <= t.t_end - t.t_start <= hi]
    lengths = [t.t_end - t.t_start for t in turns]
    print(f"== {script['id']}: {duration_s:.1f} s ({duration_s / 60:.2f} min), {len(turns)} turns, "
          f"turn length {min(lengths):.1f}-{max(lengths):.1f} s (mean {np.mean(lengths):.1f})")
    print("   talk time: " + ", ".join(f"{k} {v:.0f}s" for k, v in talk.items()))
    if outside:
        print(f"   note: {len(outside)} turn(s) outside {lo:.0f}-{hi:.0f} s: {', '.join(outside)}")
    print(f"   wrote {display_path(audio_path)} ({audio_path.stat().st_size / 1e6:.2f} MB)")


def parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        name, sep, voice = item.partition("=")
        if not sep or not name or not voice:
            raise ValueError(f"--voice expects NAME=VOICE, got {item!r}")
        out[name] = voice
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meetings", nargs="*", help="meeting ids or script paths (default: every script in demo/meetings)")
    ap.add_argument("--model-dir", type=Path, default=KOKORO_DIR)
    ap.add_argument("--download", action="store_true", help="fetch missing Kokoro model files")
    ap.add_argument("--out-dir", type=Path, default=AUDIO_DIR)
    ap.add_argument("--format", choices=("opus", "vorbis", "wav"), default="opus")
    ap.add_argument("--voice", action="append", default=[], metavar="NAME=VOICE", help="override a speaker's voice")
    ap.add_argument("--room-tone-db", type=float, default=-62.0, help="room-tone level in dBFS (-120 = off)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-truth", action="store_true", help="do not (re)write <id>.truth.json")
    args = ap.parse_args(argv)

    try:
        scripts = [load_script(p) for p in find_scripts(args.meetings)]
        overrides = parse_overrides(args.voice)
        model, voices_bin = ensure_models(args.model_dir, args.download)
        kokoro = load_kokoro(model, voices_bin)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not scripts:
        print(f"error: no meeting scripts found in {MEETINGS_DIR}", file=sys.stderr)
        return 2

    suffix = ".wav" if args.format == "wav" else ".ogg"
    for script in scripts:
        voices = voices_for(script, overrides)
        try:
            check_voices(kokoro, voices)
        except ValueError as exc:
            print(f"error: {script['id']}: {exc}", file=sys.stderr)
            return 2
        print(f"-- {script['id']}: " + ", ".join(f"{n}={v.voice}@{v.speed}" for n, v in voices.items()), flush=True)
        started = time.perf_counter()
        room_seed = args.seed + zlib.crc32(script["id"].encode())  # stable per meeting, whatever the selection
        audio_path = args.out_dir / f"{script['id']}{suffix}"
        try:
            audio, turns = render(kokoro, script, voices, args.room_tone_db, room_seed)
            write_audio(audio_path, audio, args.format)
        except (RuntimeError, OSError, sf.LibsndfileError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        duration_s = len(audio) / SAMPLE_RATE
        if not args.no_truth:
            truth_path = MEETINGS_DIR / f"{script['id']}.truth.json"
            doc = truth_document(script, voices, turns, audio_path, duration_s)
            truth_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summarise(script, turns, audio_path, duration_s)
        print(f"   synthesis took {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
