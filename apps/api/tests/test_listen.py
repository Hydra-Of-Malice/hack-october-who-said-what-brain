"""Listener tests: audio helpers, online clustering, the live session and the end-to-end fixture.

The fixture (tests/fixtures/two_speakers.wav, regenerate with tests/fixtures/build_fixture.py) is a
50 s two-voice dialogue rendered with the Windows SAPI voices David and Zira: 9 alternating turns
separated by 0.6 s of low noise. `_tight` rebuilds it with 0.15 s gaps, which the VAD cannot cut, to
exercise in-segment speaker-change detection.

Whisper-backed tests need faster-whisper-small in the HF cache; set WSW_SKIP_SLOW=1 to skip them.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from wsw_brain.brand import REPO_ROOT
from wsw_brain.config import settings
from wsw_brain.contracts import SAMPLE_RATE, ListenerCallbacks, Utterance
from wsw_brain.listen import (
    LiveSession,
    OnlineClusterer,
    WavAppender,
    decode_file,
    float_to_pcm16,
    pcm16_to_float,
    process_file,
    write_wav,
)
from wsw_brain.listen.asr import WhisperASR
from wsw_brain.listen.session import _is_garbage
from wsw_brain.listen.speakers import SherpaSpeakerEmbedder, speaker_change_points

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_WAV = FIXTURES / "two_speakers.wav"
TRUTH = json.loads((FIXTURES / "two_speakers.truth.json").read_text(encoding="utf-8"))["turns"]
MODELS = settings().models_dir
TITANET = "speaker_embedding_en.onnx"  # NeMo TitaNet-small (the default WSW_SPEAKER_MODEL)
ERES2NET = "speaker_embedding.onnx"  # 3D-Speaker ERes2Net
THRESHOLD = 0.50  # TitaNet-small is perfect from 0.45 to 0.55 on every set here; ERes2Net attributes all
# turns correctly at 0.50 too (it may open a tiny extra track on the 4-voice demo)
DEMO_TRUTH = REPO_ROOT / "demo" / "meetings" / "02-launch-sync.truth.json"  # 3 Kokoro voices (optional)
KEYWORDS = ["pricing", "dashboard", "survey", "monthly", "friday", "login", "november", "support"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class StubASR:
    """Deterministic ASR for speaker-attribution tests (speaker logic does not depend on the text)."""

    name = "stub"

    def __init__(self, fail_first: int = 0) -> None:
        self.calls = 0
        self.fail_first = fail_first

    def transcribe(self, samples: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("stub ASR exploded")
        return f"words {self.calls}", 0.9


class Collector:
    def __init__(self) -> None:
        self.utterances: list[Utterance] = []
        self.merges: list[tuple[str, str]] = []
        self.statuses: list[tuple[float, str, dict[str, Any]]] = []

    def callbacks(self) -> ListenerCallbacks:
        return ListenerCallbacks(
            on_utterance=self.utterances.append,
            on_speaker_merged=self._merged,
            on_status=lambda state, detail: self.statuses.append((time.monotonic(), state, detail)),
        )

    def _merged(self, meeting_id: str, src: str, dst: str) -> None:
        self.merges.append((src, dst))
        for u in self.utterances:
            if u.speaker == src:
                u.speaker = dst


def _need(model: str) -> Path:
    path = MODELS / model
    if not path.is_file():
        pytest.skip(f"model not downloaded: {path}")
    return path


@pytest.fixture(scope="module")
def fixture_audio() -> np.ndarray:
    return decode_file(FIXTURE_WAV)


@pytest.fixture(scope="module", params=[TITANET, ERES2NET])
def any_embedder(request: pytest.FixtureRequest) -> SherpaSpeakerEmbedder:
    return SherpaSpeakerEmbedder(_need(request.param))


@pytest.fixture(scope="module")
def titanet() -> SherpaSpeakerEmbedder:
    return SherpaSpeakerEmbedder(_need(TITANET))


@pytest.fixture(scope="module")
def whisper() -> WhisperASR:
    if os.environ.get("WSW_SKIP_SLOW") == "1":
        pytest.skip("WSW_SKIP_SLOW=1")
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(f"Systran/faster-whisper-{settings().asr_model}", "model.bin")
    if not isinstance(cached, str):
        pytest.skip(f"faster-whisper-{settings().asr_model} is not in the HF cache")
    asr = WhisperASR(settings().asr_model, "cpu", "int8", "en")
    asr.load()
    return asr


def _tight(audio: np.ndarray, gap_s: float = 0.15) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The fixture turns re-joined with `gap_s` of noise: too short for the VAD to separate them."""
    rng = np.random.default_rng(1)
    pieces = [np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)]
    cursor, turns = 0.5, []
    for turn in TRUTH:
        seg = audio[int(turn["t_start"] * SAMPLE_RATE) : int(turn["t_end"] * SAMPLE_RATE)]
        turns.append({**turn, "t_start": cursor, "t_end": cursor + seg.size / SAMPLE_RATE})
        pieces += [seg, rng.normal(0.0, 0.0008, int(gap_s * SAMPLE_RATE)).astype(np.float32)]
        cursor += seg.size / SAMPLE_RATE + gap_s
    return np.concatenate(pieces), turns


def _attribution(utterances: list[Utterance], turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Map tracks to true speakers by overlap; score time-weighted accuracy and per-turn majority."""
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    for u in utterances:
        for t in turns:
            o = min(u.t_end, t["t_end"]) - max(u.t_start, t["t_start"])
            if o > 0:
                overlap[(u.speaker, t["speaker"])] += o
    tracks = sorted({u.speaker for u in utterances}, key=lambda trk: int(trk[1:]))
    mapping: dict[str, str] = {}  # one-to-one, greedy by overlap; extra tracks count as wrong ("?")
    for (trk, sp), _ in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
        if trk not in mapping and sp not in mapping.values():
            mapping[trk] = sp
    mapping.update({trk: "?" for trk in tracks if trk not in mapping})
    total = sum(overlap.values())
    correct = sum(v for (trk, sp), v in overlap.items() if mapping[trk] == sp)
    turn_ok = 0
    for t in turns:
        votes: Counter[str] = Counter()
        for u in utterances:
            o = min(u.t_end, t["t_end"]) - max(u.t_start, t["t_start"])
            if o > 0:
                votes[mapping[u.speaker]] += o
        turn_ok += bool(votes) and votes.most_common(1)[0][0] == t["speaker"]
    return {
        "tracks": tracks,
        "mapping": mapping,
        "time_acc": correct / total if total else 0.0,
        "turn_acc": turn_ok / len(turns),
    }


def _run_session(
    audio: np.ndarray, asr: Any, embedder: Any, collector: Collector, *, live: bool = False, **kwargs: Any
) -> LiveSession:
    """Feed 100 ms PCM16 frames like the WebSocket does. live=True waits for the worker after every
    frame, i.e. the worker keeps up with the stream (each segment is handled as soon as it ends)."""
    session = LiveSession(
        "m_test",
        asr,
        embedder,
        collector.callbacks(),
        threshold=kwargs.pop("threshold", THRESHOLD),
        max_utterance_s=12.0,
        **kwargs,
    )
    pcm = float_to_pcm16(audio)
    for i in range(0, len(pcm), 3200):
        session.feed_pcm16(pcm[i : i + 3200])
        if live:
            assert session.wait_idle(120), "worker did not keep up"
    session.close(timeout=600)
    return session


def _assert_two_speakers_in_order(utterances: list[Utterance], turns: list[dict[str, Any]]) -> dict[str, Any]:
    score = _attribution(utterances, turns)
    assert score["tracks"] == ["S1", "S2"], score
    assert sorted(score["mapping"].values()) == ["David", "Zira"], score
    assert score["turn_acc"] >= 0.8, score
    seqs = [u.seq for u in utterances]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    assert all(a.t_start <= b.t_start for a, b in itertools.pairwise(utterances))
    assert all(0.0 <= u.confidence <= 1.0 and u.t_end > u.t_start for u in utterances)
    return score


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------


def test_pcm16_round_trip() -> None:
    x = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
    data = float_to_pcm16(x)
    assert len(data) == 2 * x.size
    y = pcm16_to_float(data)
    assert y.dtype == np.float32 and np.max(np.abs(y - x)) < 1e-4
    assert pcm16_to_float(float_to_pcm16(np.array([2.0, -3.0], dtype=np.float32))).tolist() == pytest.approx(
        [32767 / 32768, -32767 / 32768]
    )
    with pytest.raises(ValueError, match="odd number"):
        pcm16_to_float(b"\x00\x01\x02")


def test_decode_file_resamples_downmixes_and_reads_ogg(tmp_path: Path) -> None:
    t = np.arange(int(44_100 * 1.5)) / 44_100
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    sf.write(tmp_path / "s.wav", stereo, 44_100, subtype="PCM_16")
    sf.write(tmp_path / "s.ogg", stereo, 44_100, format="OGG", subtype="VORBIS")
    for name in ("s.wav", "s.ogg"):
        y = decode_file(tmp_path / name)
        assert y.dtype == np.float32 and y.ndim == 1
        assert abs(y.size - int(1.5 * SAMPLE_RATE)) < SAMPLE_RATE * 0.05, (name, y.size)
        assert 0.3 < float(np.sqrt(np.mean(y[1000:-1000] ** 2))) < 0.4  # 0.5 * sin has RMS 0.354
    out = write_wav(tmp_path / "nested" / "copy.wav", decode_file(tmp_path / "s.wav"))
    assert sf.info(str(out)).samplerate == SAMPLE_RATE and sf.info(str(out)).channels == 1


@pytest.mark.parametrize(
    ("fmt", "codec", "ext"),
    [("webm", "libopus", "webm"), ("ipod", "aac", "m4a"), ("mp3", "libmp3lame", "mp3")],
)
def test_decode_file_compressed_uploads(tmp_path: Path, fmt: str, codec: str, ext: str) -> None:
    import av

    t = np.arange(48_000 * 2) / 48_000
    tone = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = tmp_path / f"upload.{ext}"
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=48_000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(tone[None, :], format="fltp", layout="mono")
        frame.sample_rate = 48_000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    y = decode_file(path)
    assert abs(y.size - 2 * SAMPLE_RATE) < 0.05 * SAMPLE_RATE, y.size
    assert 0.25 < float(np.sqrt(np.mean(y[2000:-2000] ** 2))) < 0.31  # 0.4 * sin has RMS 0.283


def test_decode_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        decode_file(tmp_path / "missing.wav")
    bad = tmp_path / "bad.webm"
    bad.write_bytes(b"this is not audio" * 100)
    with pytest.raises(ValueError, match="could not decode"):
        decode_file(bad)


def test_wav_appender_is_valid_while_growing(tmp_path: Path) -> None:
    path = tmp_path / "rec" / "meeting.wav"
    rec = WavAppender(path)
    for _ in range(5):
        rec.append(np.full(1600, 0.25, dtype=np.float32))
    with wave.open(str(path), "rb") as w:  # a reader mid-meeting sees a valid header
        assert (w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
            8000,
            SAMPLE_RATE,
            1,
            2,
        )
    rec.append(np.zeros(800, dtype=np.float32))
    assert rec.duration_s == pytest.approx(0.55)
    assert rec.close() == path and rec.closed
    data, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE and data.size == 8800 and data[0] == pytest.approx(0.25, abs=1e-4)
    with pytest.raises(RuntimeError, match="closed"):
        rec.append(np.zeros(10, dtype=np.float32))
    rec.close()  # idempotent


# ---------------------------------------------------------------------------
# online clustering (synthetic vectors)
# ---------------------------------------------------------------------------


def _vec(*values: float, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


def test_clusterer_opens_new_speakers_and_assigns() -> None:
    c = OnlineClusterer(0.5)
    a, b = _vec(1, 0), _vec(0, 1)
    assert c.assign(a, 2.0) == ("S1", 1.0)
    track, conf = c.assign(b, 2.0)
    assert track == "S2" and conf == 1.0  # orthogonal to S1: clearly a new voice
    track, conf = c.assign(_vec(1, 0.2), 1.5)
    assert track == "S1" and 0.5 < conf <= 1.0
    assert c.assign(_vec(0.1, 1), 3.0)[0] == "S2"
    assert c.tracks == ["S1", "S2"] and c.previous == "S2"
    assert set(c.centroids()) == {"S1", "S2"}


def test_clusterer_short_low_similarity_segment_inherits_previous_speaker() -> None:
    c = OnlineClusterer(0.5)
    c.assign(_vec(1, 0), 2.0)
    assert c.assign(_vec(0, 1), 0.5) == ("S1", OnlineClusterer.INHERIT_CONFIDENCE)
    assert c.tracks == ["S1"]
    assert c.assign(_vec(0, 1), 2.0)[0] == "S2"  # the same voice for long enough opens a track


def test_clusterer_merges_tracks_whose_centroids_converge() -> None:
    c = OnlineClusterer(0.6)  # merge_threshold = 0.75
    assert c.merge_threshold == pytest.approx(0.75)
    c.assign(_vec(1, 0), 2.0)
    assert c.assign(_vec(0.5, 1), 2.0)[0] == "S2"  # cos = 0.447 < 0.6
    assert c.merge_close() == []
    for _ in range(6):  # S2 keeps hearing a voice between S1 and S2 (closer to S2), so it drifts to S1 ...
        assert c.assign(_vec(0.8, 0.6), 3.0)[0] == "S2"
    assert c.similarities(c.centroids()["S1"])["S2"] > 0.75
    assert c.merge_close() == [("S2", "S1")]  # ... until the newer track folds into the older one
    assert c.tracks == ["S1"] and c.canonical("S2") == "S1" and c.previous == "S1"
    assert c.assign(_vec(0, 0, 1), 2.0)[0] == "S3"  # numbers are never reused


def test_clusterer_caps_speakers_and_chooses_between_neighbours() -> None:
    c = OnlineClusterer(0.5, max_speakers=2)
    c.assign(_vec(1, 0), 2.0)
    c.assign(_vec(0, 1), 2.0)
    track, conf = c.assign(_vec(0.3, 0, 1), 2.0)  # a third voice, but the cap is 2
    assert track == "S1" and conf < 0.5 and c.tracks == ["S1", "S2"]
    assert c.best_match(_vec(0.2, 1))[0] == "S2"
    assert c.choose(_vec(0.1, 1), ["S1", "S2"])[0] == "S2"
    assert c.choose(None, ["S2", "S1"]) == ("S2", OnlineClusterer.INHERIT_CONFIDENCE)
    with pytest.raises(ValueError):
        c.choose(_vec(1, 0), ["S9"])
    with pytest.raises(ValueError):
        c.assign(np.zeros(8, dtype=np.float32), 1.0)
    with pytest.raises(ValueError):
        OnlineClusterer(1.5)


def test_garbage_filter() -> None:
    assert _is_garbage("", rms=0.1, asr_confidence=0.9, quiet_rms=0.012)
    assert _is_garbage(" ... !? ", rms=0.1, asr_confidence=0.9, quiet_rms=0.012)
    assert _is_garbage("Thank you.", rms=0.005, asr_confidence=0.9, quiet_rms=0.012)  # near-silence
    assert _is_garbage("Thank you.", rms=0.1, asr_confidence=0.2, quiet_rms=0.012)  # unsure
    assert not _is_garbage("Thank you.", rms=0.1, asr_confidence=0.8, quiet_rms=0.012)
    assert not _is_garbage("Ship it on Friday.", rms=0.005, asr_confidence=0.3, quiet_rms=0.012)


# ---------------------------------------------------------------------------
# speaker embeddings and change points (real models)
# ---------------------------------------------------------------------------


def test_embedder_is_normalised_and_separates_voices(
    any_embedder: SherpaSpeakerEmbedder, fixture_audio: np.ndarray
) -> None:
    def turn(i: int) -> np.ndarray:
        t = TRUTH[i]
        return fixture_audio[int(t["t_start"] * SAMPLE_RATE) : int(t["t_end"] * SAMPLE_RATE)]

    d1, z1, d2 = (any_embedder.embed(turn(i)) for i in (0, 1, 2))
    assert d1.shape == (any_embedder.dim,) and d1.dtype == np.float32
    assert np.linalg.norm(d1) == pytest.approx(1.0, abs=1e-5)
    assert float(d1 @ d2) > 0.6 > float(d1 @ z1)
    with pytest.raises(FileNotFoundError):
        SherpaSpeakerEmbedder(MODELS / "nope.onnx")


def test_change_point_inside_a_segment(titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray) -> None:
    david = fixture_audio[int(TRUTH[6]["t_start"] * SAMPLE_RATE) : int(TRUTH[6]["t_end"] * SAMPLE_RATE)]
    zira = fixture_audio[int(TRUTH[7]["t_start"] * SAMPLE_RATE) : int(TRUTH[7]["t_end"] * SAMPLE_RATE)]
    cuts = speaker_change_points(np.concatenate([david, zira]), titanet, THRESHOLD - 0.1)
    assert len(cuts) == 1 and abs(cuts[0] - david.size) < 0.25 * SAMPLE_RATE, (cuts, david.size)
    assert speaker_change_points(david, titanet, THRESHOLD - 0.1) == []
    assert speaker_change_points(david[:SAMPLE_RATE], titanet, THRESHOLD - 0.1) == []


# ---------------------------------------------------------------------------
# LiveSession with the real VAD and speaker models (stub ASR: fast)
# ---------------------------------------------------------------------------


def test_session_back_to_back_turns(any_embedder: SherpaSpeakerEmbedder, fixture_audio: np.ndarray) -> None:
    audio, turns = _tight(fixture_audio)
    collector = Collector()
    session = _run_session(audio, StubASR(), any_embedder, collector)
    score = _assert_two_speakers_in_order(collector.utterances, turns)
    assert score["turn_acc"] == 1.0 and score["time_acc"] > 0.95, score
    assert session.metrics()["splits"] >= 5  # most turn changes had no pause the VAD could use


def test_session_three_voice_demo_meeting(titanet: SherpaSpeakerEmbedder) -> None:
    if not DEMO_TRUTH.is_file():
        pytest.skip(f"demo meeting not generated: {DEMO_TRUTH}")
    truth = json.loads(DEMO_TRUTH.read_text(encoding="utf-8"))
    audio_path = REPO_ROOT / truth["audio"]
    if not audio_path.is_file():
        pytest.skip(f"demo audio not generated: {audio_path}")
    collector = Collector()
    _run_session(decode_file(audio_path), StubASR(), titanet, collector)
    score = _attribution(collector.utterances, truth["turns"])
    assert len(score["tracks"]) == len(truth["voices"]), score
    assert score["time_acc"] > 0.95 and score["turn_acc"] > 0.95, score


def test_session_records_audio_reports_status_and_numbers_from_start_seq(
    titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray, tmp_path: Path
) -> None:
    clip = fixture_audio[: int(13.0 * SAMPLE_RATE)]  # first two turns
    collector = Collector()
    rec = tmp_path / "audio" / "m_test.wav"
    session = _run_session(clip, StubASR(), titanet, collector, start_seq=40, recording_path=rec)
    assert collector.utterances[0].seq == 40
    assert session.next_seq == 40 + len(collector.utterances)
    assert {u.speaker for u in collector.utterances} == {"S1", "S2"}
    assert sf.info(str(rec)).frames == clip.size and session.recording_path == rec
    states = [s for _, s, _ in collector.statuses]
    assert states[0] == "listening" and states[-1] == "idle"
    assert all("lag_s" in d for _, _, d in collector.statuses)
    regular = [ts for ts, s, _ in collector.statuses if s in ("listening", "processing")]
    assert all(b - a >= 1.9 for a, b in itertools.pairwise(regular))  # at most every 2 s
    assert session.lag_s == 0.0 and not session.running
    with pytest.raises(RuntimeError, match="closed"):
        session.feed(np.zeros(160, dtype=np.float32))
    session.close()  # idempotent


def test_session_survives_asr_errors(titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray) -> None:
    collector = Collector()
    asr = StubASR(fail_first=1)
    session = _run_session(fixture_audio[: int(20 * SAMPLE_RATE)], asr, titanet, collector)
    errors = [d for _, s, d in collector.statuses if s == "error"]
    assert len(errors) == 1 and "stub ASR exploded" in errors[0]["detail"]
    assert collector.utterances and asr.calls > 1
    assert session.metrics()["errors"] == 1 and not session.running


def test_session_survives_a_failing_callback(
    titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray
) -> None:
    delivered: list[Utterance] = []
    statuses: list[tuple[str, dict[str, Any]]] = []

    def on_utterance(u: Utterance) -> None:
        if u.seq == 0:
            raise KeyError("graph writer down")
        delivered.append(u)

    callbacks = ListenerCallbacks(on_utterance=on_utterance, on_status=lambda s, d: statuses.append((s, d)))
    session = LiveSession("m_cb", StubASR(), titanet, callbacks, threshold=THRESHOLD, max_utterance_s=12.0)
    session.feed(fixture_audio[: int(20 * SAMPLE_RATE)])
    session.close()
    errors = [d["detail"] for s, d in statuses if s == "error"]
    assert errors == ["delivering an utterance: KeyError: 'graph writer down'"]
    assert [u.seq for u in delivered] == list(range(1, 1 + len(delivered))) and delivered


def test_session_ignores_silence_and_noise(titanet: SherpaSpeakerEmbedder) -> None:
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.001, 3 * SAMPLE_RATE).astype(np.float32)
    collector = Collector()
    asr = StubASR()
    session = _run_session(noise, asr, titanet, collector)
    assert collector.utterances == [] and asr.calls == 0
    assert session.metrics()["audio_s"] == pytest.approx(3.0)


def test_session_rejects_bad_config(titanet: SherpaSpeakerEmbedder, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Silero"):
        LiveSession(
            "m",
            StubASR(),
            titanet,
            Collector().callbacks(),
            threshold=0.45,
            max_utterance_s=12.0,
            vad_model_path=tmp_path / "missing.onnx",
        )
    with pytest.raises(ValueError):
        LiveSession("m", StubASR(), titanet, Collector().callbacks(), threshold=0.45, max_utterance_s=0.1)


def test_process_file_realtime_pace_and_cancel(
    titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.wav"
    write_wav(clip, fixture_audio[: int(1.5 * SAMPLE_RATE)])
    t0 = time.monotonic()
    process_file(
        clip,
        "m_rt",
        StubASR(),
        titanet,
        Collector().callbacks(),
        pace="realtime",
        threshold=THRESHOLD,
        max_utterance_s=12.0,
    )
    assert time.monotonic() - t0 >= 1.4
    cancel = threading.Event()
    cancel.set()
    assert (
        process_file(
            FIXTURE_WAV,
            "m_c",
            StubASR(),
            titanet,
            Collector().callbacks(),
            threshold=THRESHOLD,
            max_utterance_s=12.0,
            cancel=cancel,
        )
        == []
    )
    with pytest.raises(ValueError, match="pace"):
        process_file(
            clip,
            "m",
            StubASR(),
            titanet,
            Collector().callbacks(),
            pace="slow",  # type: ignore[arg-type]
            threshold=THRESHOLD,
            max_utterance_s=12.0,
        )


# ---------------------------------------------------------------------------
# End to end with faster-whisper (slow: ~10-60 s each on CPU)
# ---------------------------------------------------------------------------


def _assert_transcript(utterances: list[Utterance]) -> None:
    text = " ".join(u.text for u in utterances).lower()
    missing = [k for k in KEYWORDS if k not in text]
    assert not missing, (missing, text)


def test_process_file_fixture_with_whisper(
    whisper: WhisperASR, titanet: SherpaSpeakerEmbedder, tmp_path: Path
) -> None:
    collector = Collector()
    t0 = time.perf_counter()
    utterances = process_file(
        FIXTURE_WAV,
        "m_file",
        whisper,
        titanet,
        collector.callbacks(),
        pace="fast",
        threshold=THRESHOLD,
        max_utterance_s=12.0,
        recording_path=tmp_path / "rec.wav",
    )
    wall = time.perf_counter() - t0
    assert utterances == collector.utterances
    score = _assert_two_speakers_in_order(utterances, TRUTH)
    _assert_transcript(utterances)
    assert sf.info(str(tmp_path / "rec.wav")).duration == pytest.approx(TRUTH[-1]["t_end"], abs=1.0)
    print(f"\nprocess_file: {len(utterances)} utterances, {score}, wall {wall:.1f} s, RTF {wall / 50.2:.2f}")


def test_live_session_pcm16_frames_with_whisper(
    whisper: WhisperASR, titanet: SherpaSpeakerEmbedder, fixture_audio: np.ndarray
) -> None:
    collector = Collector()
    t0 = time.perf_counter()
    session = _run_session(fixture_audio, whisper, titanet, collector, live=True)
    wall = time.perf_counter() - t0
    score = _assert_two_speakers_in_order(collector.utterances, TRUTH)
    _assert_transcript(collector.utterances)
    print(f"\nlive: {len(collector.utterances)} utterances, {score}, wall {wall:.1f} s, {session.metrics()}")


def test_factory_builds_shared_models_from_settings() -> None:
    from wsw_brain.listen import get_asr, get_speaker_embedder

    _need(settings().speaker_model)
    assert get_asr() is get_asr() and get_asr().name == f"faster-whisper-{settings().asr_model}"
    emb = get_speaker_embedder()
    assert emb is get_speaker_embedder() and emb.dim > 0 and emb.name.startswith("sherpa-")
