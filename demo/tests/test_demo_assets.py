"""Checks on the committed demo meetings: scripts follow the contract, truth files match the audio."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import make_audio as ma
import numpy as np
import pytest
import soundfile as sf

DEMO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (DEMO / "meetings").glob("*.json") if not p.name.endswith(".truth.json"))
MAX_AUDIO_BYTES = 4_000_000


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_meetings_exist() -> None:
    assert [p.stem for p in SCRIPTS] == ["01-pricing-review", "02-launch-sync"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_contract(path: Path) -> None:
    script = ma.load_script(path)  # raises on structural problems
    assert script["id"] == path.stem
    names = {s["name"] for s in script["speakers"]}
    for spk in script["speakers"]:
        assert spk["role"] and spk["voice"]
    assert len({s["voice"] for s in script["speakers"]}) == len(names), "voices must be distinct per speaker"
    assert {t["speaker"] for t in script["turns"]} == names, "every speaker should talk"
    for a, b in itertools.pairwise(script["turns"]):
        assert a["speaker"] != b["speaker"], f"consecutive turns by {a['speaker']}: merge them"
    for t in script["turns"]:
        sentences = [s for s in re.split(r"(?<=[.?!])\s+", t["text"].strip()) if s]
        assert 1 <= len(sentences) <= 3, t["text"]
    expected = script["expected"]
    for item in expected["action_items"]:
        assert item["owner"] in {"Priya", "Arjun", "Meera", "Rahul"} and item["text"] and item["due"]
    for qa in expected["qa"]:
        assert qa["question"].endswith("?") and qa["answer_should_mention"]


def test_meeting_sizes_match_the_brief() -> None:
    first, second = (ma.load_script(p) for p in SCRIPTS)
    assert len(first["speakers"]) == 4 and 45 <= len(first["turns"]) <= 60
    assert 2 <= len(first["expected"]["decisions"]) <= 4
    assert 4 <= len(first["expected"]["action_items"]) <= 6
    assert {s["name"] for s in second["speakers"]} < {s["name"] for s in first["speakers"]}
    assert len(second["speakers"]) == 3
    assert 2 <= len(second["expected"]["action_items"]) <= 3
    assert len(second["expected"]["completed_from_previous"]) == 1
    superseding = [d for d in second["expected"]["decisions"] if d.get("supersedes")]
    assert len(superseding) == 1
    assert superseding[0]["supersedes"] in {d["text"] for d in first["expected"]["decisions"]}


def test_same_person_keeps_the_same_voice_across_meetings() -> None:
    voices: dict[str, str] = {}
    for path in SCRIPTS:
        for spk in load(path)["speakers"]:
            assert voices.setdefault(spk["name"], spk["voice"]) == spk["voice"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_truth_matches_script_and_audio(path: Path) -> None:
    truth_path = path.with_name(f"{path.stem}.truth.json")
    audio_path = DEMO / "audio" / f"{path.stem}.ogg"
    if not truth_path.exists() or not audio_path.exists():
        pytest.skip("audio not generated yet: run demo/make_audio.py")
    script, truth = load(path), load(truth_path)
    turns = truth["turns"]
    assert [(t["speaker"], t["text"]) for t in turns] == [(t["speaker"], t["text"]) for t in script["turns"]]
    for a, b in itertools.pairwise(turns):
        assert a["t_start"] < a["t_end"] <= b["t_start"]
    for t in turns:
        assert 2.0 <= t["t_end"] - t["t_start"] <= 16.0
    for prev, nxt, s in zip(turns, turns[1:], script["turns"]):
        assert nxt["t_start"] - prev["t_end"] == pytest.approx(s["pause_after_s"], abs=0.002)

    assert audio_path.stat().st_size < MAX_AUDIO_BYTES
    info = sf.info(str(audio_path))
    assert info.samplerate == ma.SAMPLE_RATE and info.channels == 1
    assert info.duration == pytest.approx(truth["duration_s"], abs=0.05)
    assert turns[-1]["t_end"] < info.duration


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_audio_has_speech_inside_turns_and_quiet_between(path: Path) -> None:
    truth_path = path.with_name(f"{path.stem}.truth.json")
    audio_path = DEMO / "audio" / f"{path.stem}.ogg"
    if not truth_path.exists() or not audio_path.exists():
        pytest.skip("audio not generated yet: run demo/make_audio.py")
    audio, rate = sf.read(str(audio_path), dtype="float32")
    turns = load(truth_path)["turns"]

    def rms_db(t0: float, t1: float) -> float:
        seg = audio[int(t0 * rate): int(t1 * rate)]
        return 20 * np.log10(max(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))), 1e-9))

    for t in turns:
        assert rms_db(t["t_start"], t["t_end"]) > -32.0
    for a, b in itertools.pairwise(turns):
        assert rms_db(a["t_end"] + 0.05, b["t_start"] - 0.05) < -50.0


def test_validation_rejects_bad_scripts() -> None:
    good = {"id": "x", "title": "x", "speakers": [{"name": "A", "voice": "af_heart"}],
            "turns": [{"speaker": "A", "text": "Hello there.", "pause_after_s": 0.5}]}
    ma.validate_script(good, "good")
    bad_pause = json.loads(json.dumps(good))
    bad_pause["turns"][0]["pause_after_s"] = 2.0
    with pytest.raises(ValueError, match="pause_after_s"):
        ma.validate_script(bad_pause, "bad")
    bad_speaker = json.loads(json.dumps(good))
    bad_speaker["turns"][0]["speaker"] = "B"
    with pytest.raises(ValueError, match="not in speakers"):
        ma.validate_script(bad_speaker, "bad")


def test_trim_and_normalise() -> None:
    rate = ma.SAMPLE_RATE
    t = np.arange(rate) / rate
    tone = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    padded = np.concatenate([np.zeros(rate // 2, np.float32), tone, np.zeros(rate // 2, np.float32)])
    trimmed = ma.trim_silence(padded)
    assert abs(len(trimmed) - rate) <= int(0.05 * rate)
    level = 20 * np.log10(np.sqrt(np.mean(ma.normalise(trimmed).astype(np.float64) ** 2)))
    assert level == pytest.approx(ma.TARGET_RMS_DBFS, abs=0.5)
    with pytest.raises(ValueError, match="silent"):
        ma.trim_silence(np.zeros(rate, np.float32))


def test_lang_follows_voice_prefix() -> None:
    assert ma.lang_for("bf_emma") == "en-gb"
    assert ma.lang_for("am_michael") == "en-us"
    assert ma.lang_for("hf_alpha") == "en-us"
