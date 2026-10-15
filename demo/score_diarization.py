"""Score the live listener's "who said what" against a demo meeting's ground truth.

  python demo/score_diarization.py demo/meetings/01-pricing-review.truth.json api_meeting.json [--json]

api_meeting.json is the body of GET /api/meetings/{id} ({"meeting", "speakers", "utterances", ...}).
A bare utterance list, or {"utterances": [...]}, works too. Fields used per utterance:
speaker (track, e.g. "S2"), speaker_name (optional), t_start, t_end, text.

Speaker metrics (computed on a 10 ms grid):
- mapping: the one-to-one track -> name assignment that maximises overlapping speech time (Hungarian).
- speaker_accuracy: of the truth speech the listener covered, the share attributed to the right person.
- der: (missed + false alarm + confusion) / truth speech, ignoring +-collar around truth turn boundaries.
- turn_accuracy: truth turns whose dominant track maps to the right person.
- utterance_accuracy: listener utterances whose mapped person is the majority truth speaker under them.
- named_accuracy: like speaker_accuracy, but trusting the names given in the UI (speaker_name).
Text metrics (lower-cased, punctuation stripped, number words -> digits, currency words dropped):
- wer: word error rate of the time-ordered transcript; turn_wer: mean WER per truth turn;
  word_recall: share of each truth turn's words found in the utterances that overlap it most.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STEP_S = 0.01
NONE = -1


@dataclass(frozen=True)
class Segment:
    label: str  # truth: person name; listener: speaker track
    t_start: float
    t_end: float
    text: str
    name: str | None = None  # listener only: speaker_name assigned in the UI


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _segment(row: dict[str, Any], label_key: str, where: str) -> Segment:
    try:
        t0, t1 = float(row["t_start"]), float(row["t_end"])
        label = str(row[label_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: needs '{label_key}', 't_start' and 't_end' ({exc})") from exc
    if t1 < t0:
        raise ValueError(f"{where}: t_end {t1} is before t_start {t0}")
    return Segment(label, t0, t1, str(row.get("text") or ""), row.get("speaker_name") or None)


def load_truth(data: Any) -> list[Segment]:
    turns = data.get("turns") if isinstance(data, dict) else data
    if not isinstance(turns, list) or not turns:
        raise ValueError("truth: expected a non-empty list of turns (or {'turns': [...]})")
    return sorted((_segment(t, "speaker", f"truth turn {i}") for i, t in enumerate(turns)), key=lambda s: s.t_start)


def load_hypothesis(data: Any) -> list[Segment]:
    rows = data.get("utterances") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise TypeError("listener: expected GET /api/meetings/{id} JSON, {'utterances': [...]} or a list")
    segs = [_segment(u, "speaker", f"utterance {i}") for i, u in enumerate(rows)]
    return sorted((s for s in segs if s.t_end > s.t_start), key=lambda s: s.t_start)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON ({exc})") from exc


# ---------------------------------------------------------------------------
# Assignment (Hungarian algorithm, shortest augmenting path)
# ---------------------------------------------------------------------------


def min_cost_assignment(cost: np.ndarray) -> list[int]:
    """Square cost matrix -> column chosen for each row, minimising the total cost."""
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError(f"cost matrix must be square, got {cost.shape}")
    row_pot = np.zeros(n + 1)
    col_pot = np.zeros(n + 1)
    owner = np.zeros(n + 1, dtype=int)  # owner[j] = row (1-based) matched to column j; column 0 is a sentinel
    for row in range(1, n + 1):
        owner[0] = row
        col = 0
        best = np.full(n + 1, math.inf)
        prev = np.zeros(n + 1, dtype=int)
        used = np.zeros(n + 1, dtype=bool)
        while owner[col] != 0:
            used[col] = True
            r = owner[col]
            delta, nxt = math.inf, 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                reduced = cost[r - 1, j - 1] - row_pot[r] - col_pot[j]
                if reduced < best[j]:
                    best[j], prev[j] = reduced, col
                if best[j] < delta:
                    delta, nxt = best[j], j
            for j in range(n + 1):
                if used[j]:
                    row_pot[owner[j]] += delta
                    col_pot[j] -= delta
                else:
                    best[j] -= delta
            col = nxt
        while col:
            back = prev[col]
            owner[col] = owner[back]
            col = back
    result = [0] * n
    for j in range(1, n + 1):
        result[owner[j] - 1] = j - 1
    return result


def best_mapping(overlap: np.ndarray, tracks: list[str], names: list[str]) -> dict[str, str]:
    """Track -> name, one-to-one, maximising total overlap. Tracks with no overlap stay unmapped."""
    size = max(len(tracks), len(names))
    if size == 0:
        return {}
    square = np.zeros((size, size))
    square[: len(tracks), : len(names)] = overlap
    cols = min_cost_assignment(-square)
    return {tracks[r]: names[c] for r, c in enumerate(cols)
            if r < len(tracks) and c < len(names) and overlap[r, c] > 0}


# ---------------------------------------------------------------------------
# Speaker scoring
# ---------------------------------------------------------------------------


def _frames(t: float) -> int:
    return round(t / STEP_S)


def paint(segments: list[Segment], labels: list[str], n_frames: int) -> np.ndarray:
    """Frame grid of label indices (NONE where nobody speaks). Later segments win on overlap."""
    index = {lab: i for i, lab in enumerate(labels)}
    grid = np.full(n_frames, NONE, dtype=int)
    for s in segments:
        grid[_frames(s.t_start): _frames(s.t_end)] = index[s.label]
    return grid


def collar_mask(truth: list[Segment], n_frames: int, collar_s: float) -> np.ndarray:
    scored = np.ones(n_frames, dtype=bool)
    c = _frames(collar_s)
    if c <= 0:
        return scored
    for s in truth:
        for edge in (_frames(s.t_start), _frames(s.t_end)):
            scored[max(0, edge - c): edge + c] = False
    return scored


def dominant(values: np.ndarray) -> int:
    values = values[values != NONE]
    if values.size == 0:
        return NONE
    return int(np.bincount(values).argmax())


def named_accuracy(hyp: list[Segment], names: list[str], tracks: list[str], ref: np.ndarray, out: np.ndarray,
                   both: np.ndarray) -> dict[str, Any] | None:
    """Accuracy when trusting the names typed in the UI (case-insensitive); unnamed tracks count as wrong."""
    by_lower = {n.lower(): i for i, n in enumerate(names)}
    named = {s.label: s.name.strip() for s in hyp if s.name and s.name.strip()}
    if not named:
        return None
    lookup = np.array([by_lower.get(named.get(t, "").lower(), NONE) for t in tracks] + [NONE])
    hit = both & (lookup[out] == ref)
    return {"accuracy": float(hit.sum() / max(both.sum(), 1)), "named_tracks": len(named), "tracks": len(tracks),
            "names": named}


def score_speakers(truth: list[Segment], hyp: list[Segment], collar_s: float = 0.25) -> dict[str, Any]:
    names = sorted({s.label for s in truth})
    tracks = sorted({s.label for s in hyp}, key=lambda t: (len(t), t))
    end = max([s.t_end for s in truth] + [s.t_end for s in hyp])
    n = _frames(end) + 1
    ref = paint(truth, names, n)
    out = paint(hyp, tracks, n)
    scored = collar_mask(truth, n, collar_s)

    overlap = np.zeros((len(tracks), len(names)))
    both = scored & (ref != NONE) & (out != NONE)
    np.add.at(overlap, (out[both], ref[both]), STEP_S)
    mapping = best_mapping(overlap, tracks, names)
    to_name = np.array([names.index(mapping[t]) if t in mapping else NONE for t in tracks] + [NONE])

    speech = scored & (ref != NONE)
    missed = speech & (out == NONE)
    false_alarm = scored & (ref == NONE) & (out != NONE)
    confusion = both & (to_name[out] != ref)
    correct = both & (to_name[out] == ref)
    speech_s = speech.sum() * STEP_S

    turn_hits = sum(1 for s in truth if (d := dominant(out[_frames(s.t_start): _frames(s.t_end)])) != NONE
                    and mapping.get(tracks[d]) == s.label)
    utt_rows = []
    for s in hyp:
        major = dominant(ref[_frames(s.t_start): _frames(s.t_end)])
        if major != NONE:
            utt_rows.append(mapping.get(s.label) == names[major])

    return {
        "names": names,
        "tracks": tracks,
        "mapping": mapping,
        "overlap_s": {t: {nm: round(float(overlap[i, j]), 2) for j, nm in enumerate(names) if overlap[i, j] > 0}
                      for i, t in enumerate(tracks)},
        "speaker_accuracy": float(correct.sum() / max(both.sum(), 1)),
        "coverage": float(both.sum() / max(speech.sum(), 1)),
        "der": float((missed.sum() + false_alarm.sum() + confusion.sum()) / max(speech.sum(), 1)),
        "missed_s": round(float(missed.sum() * STEP_S), 2),
        "false_alarm_s": round(float(false_alarm.sum() * STEP_S), 2),
        "confusion_s": round(float(confusion.sum() * STEP_S), 2),
        "truth_speech_s": round(float(speech_s), 2),
        "turns": len(truth),
        "turn_correct": turn_hits,
        "turn_accuracy": turn_hits / len(truth),
        "utterances": len(hyp),
        "utterance_correct": sum(utt_rows),
        "utterance_accuracy": (sum(utt_rows) / len(utt_rows)) if utt_rows else 0.0,
        "named": named_accuracy(hyp, names, tracks, ref, out, both),
        "collar_s": collar_s,
    }


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------

UNITS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
])}
TENS = {w: 10 * (i + 2) for i, w in enumerate(
    ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])}
SCALES = {"thousand": 1_000, "lakh": 100_000, "million": 1_000_000, "crore": 10_000_000}
ORDINALS = {w: i + 1 for i, w in enumerate([
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
    "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth",
])}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"}
CURRENCY = {"rupee", "rupees", "rs", "inr"}


def _is_number_word(tok: str) -> bool:
    return tok in UNITS or tok in TENS or tok == "hundred" or tok in SCALES


def _numbers_to_digits(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if not _is_number_word(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue
        total = current = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "and" and i + 1 < len(tokens) and _is_number_word(tokens[i + 1]) and (total or current):
                i += 1
                continue
            if tok in UNITS or tok in TENS:
                current += UNITS.get(tok, TENS.get(tok, 0))
            elif tok == "hundred":
                current = max(current, 1) * 100
            elif tok in SCALES:
                total += max(current, 1) * SCALES[tok]
                current = 0
            else:
                break
            i += 1
        out.append(str(total + current))
    return out


def normalise_text(text: str) -> list[str]:
    t = text.lower().replace("₹", " rupees ").replace("%", " percent ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"[^a-z0-9.' ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t).replace("'", "")
    tokens = _numbers_to_digits(t.split())
    out: list[str] = []
    for tok in tokens:
        if tok in CURRENCY:
            continue
        if out and out[-1] in MONTHS and tok in ORDINALS:
            tok = str(ORDINALS[tok])
        out.append(tok)
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def word_recall(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0
    have = Counter(hyp)
    return sum(min(c, have[w]) for w, c in Counter(ref).items()) / len(ref)


def score_text(truth: list[Segment], hyp: list[Segment]) -> dict[str, Any]:
    ref_all = [w for s in truth for w in normalise_text(s.text)]
    hyp_all = [w for s in hyp for w in normalise_text(s.text)]
    per_turn: dict[int, list[str]] = {i: [] for i in range(len(truth))}
    for s in hyp:
        overlaps = [min(s.t_end, t.t_end) - max(s.t_start, t.t_start) for t in truth]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0:
            per_turn[best] += normalise_text(s.text)
    turn_wers = [wer(normalise_text(t.text), per_turn[i]) for i, t in enumerate(truth)]
    recalls = [word_recall(normalise_text(t.text), per_turn[i]) for i, t in enumerate(truth)]
    return {
        "wer": wer(ref_all, hyp_all),
        "turn_wer": float(np.mean(turn_wers)),
        "word_recall": float(np.mean(recalls)),
        "ref_words": len(ref_all),
        "hyp_words": len(hyp_all),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def score(truth: list[Segment], hyp: list[Segment], collar_s: float = 0.25) -> dict[str, Any]:
    if not hyp:
        raise ValueError("listener produced no utterances to score")
    return {"speakers": score_speakers(truth, hyp, collar_s), "text": score_text(truth, hyp)}


def _pct(x: float) -> str:
    return f"{100.0 * x:5.1f} %"


def report(result: dict[str, Any], title: str) -> str:
    sp, tx = result["speakers"], result["text"]
    mapped = [f"{t} -> {sp['mapping'][t]}" if t in sp["mapping"] else f"{t} -> (unmapped)" for t in sp["tracks"]]
    lines = [
        f"Diarization score: {title}",
        (f"  truth          {sp['turns']} turns, {len(sp['names'])} speakers ({', '.join(sp['names'])}), "
         f"{sp['truth_speech_s']:.1f} s of scored speech"),
        f"  listener       {sp['utterances']} utterances, {len(sp['tracks'])} tracks",
        f"  mapping        {', '.join(mapped)}",
        f"  speaker acc    {_pct(sp['speaker_accuracy'])}  (time-weighted, of covered speech)",
        f"  turn acc       {_pct(sp['turn_accuracy'])}  ({sp['turn_correct']}/{sp['turns']} truth turns)",
        f"  utterance acc  {_pct(sp['utterance_accuracy'])}  ({sp['utterance_correct']}/{sp['utterances']} utterances)",
        (f"  DER            {_pct(sp['der'])}  (missed {sp['missed_s']} s, false alarm {sp['false_alarm_s']} s, "
         f"confusion {sp['confusion_s']} s; collar {sp['collar_s']} s)"),
        f"  coverage       {_pct(sp['coverage'])}  of truth speech",
    ]
    if sp["named"]:
        nm = sp["named"]
        lines.append(f"  named acc      {_pct(nm['accuracy'])}  (using UI names; {nm['named_tracks']}/{nm['tracks']} "
                     f"tracks named)")
    lines.append(f"  text           WER {_pct(tx['wer'])}, mean turn WER {_pct(tx['turn_wer'])}, "
                 f"word recall {_pct(tx['word_recall'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", type=Path, help="demo/meetings/<id>.truth.json")
    ap.add_argument("listener", type=Path, help="GET /api/meetings/{id} response saved as JSON")
    ap.add_argument("--collar", type=float, default=0.25, help="seconds ignored around truth turn edges (DER)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--min-speaker-acc", type=float, default=None,
                    help="exit 1 if speaker accuracy is below this fraction (e.g. 0.9)")
    args = ap.parse_args(argv)
    try:
        truth_doc = read_json(args.truth)
        truth = load_truth(truth_doc)
        hyp = load_hypothesis(read_json(args.listener))
        result = score(truth, hyp, args.collar)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    title = truth_doc.get("meeting_id", args.truth.stem) if isinstance(truth_doc, dict) else args.truth.stem
    print(json.dumps(result, indent=2) if args.json else report(result, title))
    if args.min_speaker_acc is not None and result["speakers"]["speaker_accuracy"] < args.min_speaker_acc:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
