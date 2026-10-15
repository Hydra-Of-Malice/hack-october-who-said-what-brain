"""Seed the demo meetings (demo/meetings/*.json + demo/audio/*) through the REAL listener.

Each meeting is diarized and transcribed by the same LiveSession a microphone uses, written live into
FalkorDB and extracted by Gemma 4. Speakers are then named from the script's ground truth: each track
gets the truth speaker it overlaps most in time. Meetings are seeded in file order, and each one is fully
extracted before the next starts, because a later meeting can supersede an earlier decision.

Demo meetings get stable ids ("m_" + 10 hex of sha1), so an interrupted seed (still "live" at startup) is
detected, deleted and seeded again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..brand import REPO_ROOT
from ..contracts import Utterance

if TYPE_CHECKING:
    from .services import Services

log = logging.getLogger(__name__)

DEMO_DIR = REPO_ROOT / "demo"
AUDIO_SUFFIXES = (".ogg", ".wav", ".opus", ".flac", ".mp3", ".m4a")
MIN_NAME_SHARE = 0.5  # a track is named only when most of its speech overlaps one truth speaker
_DATE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})(?:,?\s+(\d{1,2}):(\d{2}))?(?:\s+(IST|UTC|GMT))?",
    re.IGNORECASE,
)
_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_ZONES = {"ist": timedelta(hours=5, minutes=30), "utc": timedelta(0), "gmt": timedelta(0)}


@dataclass(frozen=True)
class DemoMeeting:
    key: str  # file stem, e.g. "01-pricing-review"
    title: str
    audio: Path
    truth: Path | None
    started_at_ms: int | None

    @property
    def meeting_id(self) -> str:
        return demo_meeting_id(self.key)


def demo_meeting_id(key: str) -> str:
    return "m_" + hashlib.sha1(f"demo:{key}".encode()).hexdigest()[:10]


def find_demo_meetings(root: Path = DEMO_DIR) -> list[DemoMeeting]:
    """Scripts in root/meetings that have audio in root/audio, in file-name order."""
    meetings: list[DemoMeeting] = []
    for script in sorted((root / "meetings").glob("*.json")):
        if script.name.endswith(".truth.json"):
            continue
        audio = next(
            (
                root / "audio" / f"{script.stem}{s}"
                for s in AUDIO_SUFFIXES
                if (root / "audio" / f"{script.stem}{s}").is_file()
            ),
            None,
        )
        if audio is None:
            continue
        try:
            data = json.loads(script.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("skipping demo script %s: %s", script.name, exc)
            continue
        truth = script.with_name(f"{script.stem}.truth.json")
        meetings.append(
            DemoMeeting(
                key=script.stem,
                title=str(data.get("title") or script.stem),
                audio=audio,
                truth=truth if truth.is_file() else None,
                started_at_ms=parse_date_hint(str(data.get("date_hint") or "")),
            )
        )
    return meetings


def parse_date_hint(text: str) -> int | None:
    """'Tuesday 13 October 2026, 11:00 IST' -> epoch ms (IST assumed when no zone is given)."""
    match = _DATE.search(text)
    if not match:
        return None
    day, month, year, hour, minute, zone = match.groups()
    offset = _ZONES[(zone or "ist").lower()]
    try:
        moment = datetime(
            int(year),
            _MONTHS.index(month.lower()) + 1,
            int(day),
            int(hour or 10),
            int(minute or 0),
            tzinfo=timezone(offset),
        )
    except ValueError:
        return None
    return int(moment.timestamp() * 1000)


def plan_seed(
    existing: Sequence[dict[str, Any]], demos: Sequence[DemoMeeting]
) -> list[tuple[DemoMeeting, bool]]:
    """[(demo, delete_first)]. An empty graph gets every demo meeting. Otherwise only a seed that was
    interrupted (a demo meeting still live) is repaired: it is redone, with any demo meetings missing."""
    if not existing:
        return [(demo, False) for demo in demos]
    by_id = {m["id"]: m for m in existing}
    interrupted = {d.meeting_id for d in demos if by_id.get(d.meeting_id, {}).get("status") == "live"}
    if not interrupted:
        return []
    return [
        (d, d.meeting_id in interrupted)
        for d in demos
        if d.meeting_id in interrupted or d.meeting_id not in by_id
    ]


def load_truth_turns(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        turns = json.loads(path.read_text(encoding="utf-8")).get("turns") or []
    except (OSError, ValueError) as exc:
        log.warning("could not read demo truth %s: %s", path, exc)
        return []
    return [t for t in turns if isinstance(t, dict) and t.get("speaker")]


def name_tracks(utterances: Sequence[Utterance], truth_turns: Sequence[dict[str, Any]]) -> dict[str, str]:
    """track -> truth speaker name with the largest time overlap (when it covers MIN_NAME_SHARE of the
    track's overlapped speech)."""
    overlap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for u in utterances:
        for turn in truth_turns:
            shared = min(u.t_end, float(turn["t_end"])) - max(u.t_start, float(turn["t_start"]))
            if shared > 0:
                overlap[u.speaker][str(turn["speaker"])] += shared
    names: dict[str, str] = {}
    for track, by_name in overlap.items():
        name, best = max(by_name.items(), key=lambda kv: kv[1])
        if best >= MIN_NAME_SHARE * sum(by_name.values()):
            names[track] = name
    return names


def seed_demo(services: Services, cancel: threading.Event) -> None:
    """Thread target: seed (or repair) the demo meetings when the graph needs them."""
    demos = find_demo_meetings()
    if not demos:
        log.info("demo seed: no demo meetings with audio under %s", DEMO_DIR)
        return
    if not services.ensure_graph():
        log.warning("demo seed skipped: FalkorDB is not ready (%s)", services.graph_error)
        return
    plan = plan_seed(services.store.list_meetings(), demos)
    if not plan:
        log.info("demo seed: graph '%s' already has meetings; nothing to seed", services.settings.graph)
        return
    log.info("demo seed: ingesting %s through the live listener", ", ".join(d.key for d, _ in plan))
    for demo, delete_first in plan:
        if cancel.is_set():
            return
        if delete_first:
            services.call(services.delete_meeting(demo.meeting_id))
        seed_meeting(services, demo, cancel)


class _EarlyNamer:
    """Names a track from the ground truth as soon as its first utterance arrives, so live extraction
    already reads "Priya: ..." rather than "Speaker 1: ..." (and writes "Priya will draft the page")."""

    def __init__(self, services: Services, meeting_id: str, truth_turns: Sequence[dict[str, Any]]) -> None:
        self.services = services
        self.meeting_id = meeting_id
        self.truth_turns = truth_turns
        self.named: set[str] = set()

    def on_utterance(self, u: Utterance) -> None:  # listener thread, after the pipeline queued `u`
        if u.speaker in self.named:
            return
        name = name_tracks([u], self.truth_turns).get(u.speaker)
        if name is None:
            return
        self.named.add(u.speaker)
        self.services.pipeline.flush(timeout=15.0)  # the Speaker node exists once the utterance is written
        try:
            self.services.name_speaker(self.meeting_id, u.speaker, name)
        except LookupError as exc:
            log.info("demo seed: could not name %s yet: %s", u.speaker, exc)
            self.named.discard(u.speaker)


def seed_meeting(services: Services, demo: DemoMeeting, cancel: threading.Event) -> None:
    t0 = time.perf_counter()
    mid = demo.meeting_id
    services.new_meeting(demo.title, "demo", meeting_id=mid, started_at_ms=demo.started_at_ms)
    truth = load_truth_turns(demo.truth)
    namer = _EarlyNamer(services, mid, truth)
    utterances, duration = services.files.run_file(
        mid, demo.audio, pace="fast", cancel=cancel, on_utterance=namer.on_utterance
    )
    if cancel.is_set():
        log.info("demo seed of %s interrupted; it is redone at the next start", demo.key)
        return
    services.pipeline.flush()
    named = name_tracks(utterances, truth)  # the final word: majority over the whole meeting
    for track, name in sorted(named.items()):
        try:
            services.name_speaker(mid, track, name)
        except LookupError:  # the track was merged away after the utterances were collected
            log.info("demo seed %s: speaker %s no longer exists; not naming it %s", demo.key, track, name)
    listened_s = time.perf_counter() - t0
    ended_at = demo.started_at_ms + int(duration * 1000) if demo.started_at_ms and duration else None
    services.files.end_meeting(mid, duration, wait_final=True, ended_at_ms=ended_at)  # the script's clock
    log.info(
        "demo seed %s: %d utterances, %d tracks named %s, listened in %.0f s, extracted and summarised in %.0f s",
        demo.key,
        len(utterances),
        len(named),
        named,
        listened_s,
        time.perf_counter() - t0 - listened_s,
    )
