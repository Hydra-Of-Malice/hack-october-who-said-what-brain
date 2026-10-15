"""Speaker embeddings (sherpa-onnx) and online clustering into meeting-scoped tracks "S1", "S2", ..."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import SAMPLE_RATE, SpeakerEmbedder
from ._sherpa import import_sherpa
from .audio import as_mono_float32


class SherpaSpeakerEmbedder:
    """Voice embeddings from a sherpa-onnx speaker model (NeMo TitaNet, 3D-Speaker ERes2Net, ...).

    Implements contracts.SpeakerEmbedder. One instance can be shared by sessions; calls are serialised.
    """

    def __init__(self, model_path: str | Path, num_threads: int = 2) -> None:
        sherpa_onnx = import_sherpa()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"speaker embedding model not found: {path} (run scripts/fetch_models.py or set WSW_SPEAKER_MODEL)"
            )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=num_threads)
        if not config.validate():
            raise ValueError(f"invalid sherpa-onnx speaker embedding config for {path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._lock = threading.Lock()
        self.model_path = path
        self.name = f"sherpa-{path.stem}"
        self.dim = int(self._extractor.dim)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        audio = as_mono_float32(samples)
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                raise ValueError(
                    f"segment too short for a speaker embedding ({audio.size / SAMPLE_RATE:.2f} s)"
                )
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("speaker model returned a zero or non-finite embedding")
        return vec / norm


@dataclass
class _Track:
    track: str
    total: np.ndarray  # duration-weighted sum of embeddings
    weight: float
    segments: int

    @property
    def centroid(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.total))
        return self.total / norm if norm > 0 else self.total


class OnlineClusterer:
    """Greedy online speaker clustering against duration-weighted running centroids.

    - best cosine similarity >= threshold: join that track and update its centroid;
    - below threshold: open a new track (until max_speakers; then the closest track takes it);
    - a short segment (< short_s) that matches nothing inherits the previous speaker, because short
      clips give noisy embeddings and turn-taking makes "same as before" the better prior;
    - `merge_close()` folds tracks whose centroids converged (sim > merge_threshold) into the older one.

    Tracks are numbered by first appearance and numbers are never reused after a merge.
    """

    INHERIT_CONFIDENCE = 0.35

    def __init__(
        self,
        threshold: float,
        max_speakers: int = 8,
        merge_threshold: float | None = None,
        *,
        short_s: float = 0.8,
        max_weight_s: float = 10.0,
    ) -> None:
        if not -1.0 < threshold < 1.0:
            raise ValueError(f"threshold must be a cosine similarity in (-1, 1), got {threshold}")
        if max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.merge_threshold = min(0.99, threshold + 0.15) if merge_threshold is None else merge_threshold
        self.short_s = short_s
        self.max_weight_s = max_weight_s
        self._tracks: dict[str, _Track] = {}
        self._aliases: dict[str, str] = {}  # merged-away track -> track it was merged into
        self._next = 1
        self._previous: str | None = None

    # ---- queries -----------------------------------------------------------------
    @property
    def tracks(self) -> list[str]:
        return list(self._tracks)

    @property
    def previous(self) -> str | None:
        return self._previous

    def canonical(self, track: str) -> str:
        """The live track a (possibly merged-away) track now belongs to."""
        while track in self._aliases:
            track = self._aliases[track]
        return track

    def centroids(self) -> dict[str, np.ndarray]:
        return {name: t.centroid.copy() for name, t in self._tracks.items()}

    def similarities(self, embedding: np.ndarray) -> dict[str, float]:
        emb = _unit(embedding)
        return {name: float(np.dot(emb, t.centroid)) for name, t in self._tracks.items()}

    def best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """(closest track, cosine similarity), or (None, -1.0) before the first speaker exists."""
        sims = self.similarities(embedding)
        if not sims:
            return None, -1.0
        track = max(sims, key=sims.__getitem__)
        return track, sims[track]

    def choose(self, embedding: np.ndarray | None, candidates: list[str]) -> tuple[str, float]:
        """Pick the closest of `candidates` without updating any centroid (for short, ambiguous
        segments resolved by their neighbours). Without an embedding the first candidate wins."""
        live = list(dict.fromkeys(self.canonical(c) for c in candidates if self.canonical(c) in self._tracks))
        if not live:
            raise ValueError(f"none of the candidate tracks exist: {candidates}")
        if embedding is None or len(live) == 1:
            return live[0], self.INHERIT_CONFIDENCE
        sims = self.similarities(embedding)
        ranked = sorted(live, key=sims.__getitem__, reverse=True)
        margin = sims[ranked[0]] - sims[ranked[1]]
        return ranked[0], round(float(np.clip(self.INHERIT_CONFIDENCE + margin, 0.0, 0.7)), 3)

    # ---- updates -----------------------------------------------------------------
    def assign(self, embedding: np.ndarray, duration_s: float) -> tuple[str, float]:
        emb = _unit(embedding)
        sims = self.similarities(emb)
        if not sims:
            return self._open(emb, duration_s), 1.0
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best, best_sim = ranked[0]
        if best_sim >= self.threshold:
            runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
            self._update(best, emb, duration_s)
            return best, _match_confidence(best_sim, runner_up, self.threshold)
        if duration_s < self.short_s and self._previous is not None:
            return self._previous, self.INHERIT_CONFIDENCE
        if len(self._tracks) >= self.max_speakers:
            self._previous = best
            return best, round(max(0.0, best_sim) / self.threshold * 0.5, 3)
        return self._open(emb, duration_s), _new_track_confidence(best_sim, self.threshold)

    def merge_close(self) -> list[tuple[str, str]]:
        """Merge every pair of tracks whose centroids are closer than merge_threshold.

        Returns [(merged_away, kept)] in the order applied; the kept track is the older one.
        """
        merges: list[tuple[str, str]] = []
        while True:
            pair = self._closest_pair()
            if pair is None:
                return merges
            keep, drop = pair
            kept, gone = self._tracks[keep], self._tracks.pop(drop)
            kept.total = kept.total + gone.total
            kept.weight += gone.weight
            kept.segments += gone.segments
            self._aliases[drop] = keep
            if self._previous == drop:
                self._previous = keep
            merges.append((drop, keep))

    def _closest_pair(self) -> tuple[str, str] | None:
        names = list(self._tracks)  # insertion order == first appearance
        best: tuple[float, str, str] | None = None
        for i, a in enumerate(names):
            ca = self._tracks[a].centroid
            for b in names[i + 1 :]:
                sim = float(np.dot(ca, self._tracks[b].centroid))
                if sim > self.merge_threshold and (best is None or sim > best[0]):
                    best = (sim, a, b)
        return None if best is None else (best[1], best[2])

    def _open(self, emb: np.ndarray, duration_s: float) -> str:
        name = f"S{self._next}"
        self._next += 1
        weight = self._weight(duration_s)
        self._tracks[name] = _Track(track=name, total=emb * weight, weight=weight, segments=1)
        self._previous = name
        return name

    def _update(self, name: str, emb: np.ndarray, duration_s: float) -> None:
        t = self._tracks[name]
        weight = self._weight(duration_s)
        t.total = t.total + emb * weight
        t.weight += weight
        t.segments += 1
        self._previous = name

    def _weight(self, duration_s: float) -> float:
        return float(min(max(duration_s, 0.1), self.max_weight_s))


def speaker_change_points(
    samples: np.ndarray,
    embedder: SpeakerEmbedder,
    threshold: float,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    probe_s: float = 0.5,
    probe_hop_s: float = 0.1,
    min_piece_s: float = 0.3,
    max_changes: int = 3,
) -> list[int]:
    """Sample offsets where the voice changes inside one VAD segment.

    When two people talk back to back with less silence than the VAD needs to cut, both end up in one
    segment. Coarse pass: embed 1 s windows (the last one aligned to the segment end so a short reply
    at the end is covered) and pick the window edge whose left and right mean embeddings are least
    similar; below `threshold` that is a change. Fine pass: classify short probes around it as
    "left voice" or "right voice" and cut where the labels flip, snapped to the quietest 20 ms nearby.
    Recurses on both sides. Quiet windows (pauses) are skipped so they cannot fake a change.
    """
    audio = as_mono_float32(samples)
    n, win, hop = audio.size, int(window_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    min_piece = int(min_piece_s * SAMPLE_RATE)
    if n < win + hop:
        return []
    floor = 0.3 * _rms(audio)
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)
    grid_list, vectors = [], []
    for s in starts:
        vec = _embed_or_none(embedder, audio[s : s + win], floor)
        if vec is not None:
            grid_list.append(s)
            vectors.append(vec)
    if len(grid_list) < 2:
        return []
    grid, emb = np.asarray(grid_list), np.stack(vectors)
    cuts: list[int] = []

    def split(lo: int, hi: int) -> None:
        if len(cuts) >= max_changes:
            return
        inside = (grid >= lo) & (grid + win <= hi)
        edges = sorted({int(g) for g in grid[inside]} | {int(g) + win for g in grid[inside]})
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for b in edges:
            left, right = inside & (grid + win <= b), inside & (grid >= b)
            if b - lo < min_piece or hi - b < min_piece or not left.any() or not right.any():
                continue
            ref_l, ref_r = _unit(emb[left].sum(axis=0)), _unit(emb[right].sum(axis=0))
            score = float(np.dot(ref_l, ref_r))
            if best is None or score < best[0]:
                best = (score, b, ref_l, ref_r)
        if best is None or best[0] >= threshold:
            return
        cut = _refine_cut(
            audio,
            embedder,
            best[1],
            best[2],
            best[3],
            lo=lo + min_piece,
            hi=hi - min_piece,
            probe=int(probe_s * SAMPLE_RATE),
            step=int(probe_hop_s * SAMPLE_RATE),
            floor=floor,
        )
        cuts.append(cut)
        split(lo, cut)
        split(cut, hi)

    split(0, n)
    return sorted(cuts)


def _refine_cut(
    audio: np.ndarray,
    embedder: SpeakerEmbedder,
    coarse: int,
    ref_l: np.ndarray,
    ref_r: np.ndarray,
    *,
    lo: int,
    hi: int,
    probe: int,
    step: int,
    floor: float,
) -> int:
    """Best step-function fit of "sounds like the right side" scores of short probes near `coarse`."""
    a, z = max(lo, coarse - SAMPLE_RATE), min(hi, coarse + SAMPLE_RATE)
    if z - a < step:
        return int(min(max(coarse, lo), hi))
    centers = list(range(a, z + 1, step))
    scores = []
    for c in centers:
        p0 = max(0, min(c - probe // 2, audio.size - probe))
        vec = _embed_or_none(embedder, audio[p0 : p0 + probe], floor)
        scores.append(0.0 if vec is None else float(np.dot(vec, ref_r) - np.dot(vec, ref_l)))
    cum = np.concatenate(([0.0], np.cumsum(scores)))
    k = int(np.argmax(cum[-1] - 2.0 * cum))  # left of k should score negative, right of k positive
    if k == 0:
        cut = a
    elif k == len(centers):
        cut = z
    else:
        cut = (centers[k - 1] + centers[k]) // 2
    return _quietest(audio, cut, radius=int(0.15 * SAMPLE_RATE), lo=lo, hi=hi)


def _embed_or_none(embedder: SpeakerEmbedder, window: np.ndarray, floor: float) -> np.ndarray | None:
    if _rms(window) < floor:
        return None
    try:
        return embedder.embed(window)
    except ValueError:
        return None


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def _quietest(audio: np.ndarray, center: int, *, radius: int, lo: int, hi: int, frame: int = 320) -> int:
    """Middle of the lowest-energy `frame` (20 ms) within center +- radius, kept inside [lo, hi]."""
    a, b = max(lo, center - radius), min(hi, center + radius)
    if b - a < frame:
        return int(min(max(center, lo), hi))
    frames = audio[a : a + ((b - a) // frame) * frame].reshape(-1, frame)
    return a + int(np.argmin(np.square(frames).mean(axis=1))) * frame + frame // 2


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot cluster a zero or non-finite embedding")
    return arr / norm


def _match_confidence(sim: float, runner_up: float, threshold: float) -> float:
    """0.5 at the threshold rising to 1.0 at sim 1; scaled down when another track is nearly as close."""
    base = 0.5 + 0.5 * (sim - threshold) / max(1e-6, 1.0 - threshold)
    margin = sim - runner_up
    ambiguity = min(1.0, max(0.0, margin / 0.1))  # full credit once the runner-up is 0.1 behind
    return round(float(np.clip(base * (0.6 + 0.4 * ambiguity), 0.0, 1.0)), 3)


def _new_track_confidence(best_sim: float, threshold: float) -> float:
    """How clearly the segment differs from every known speaker: 0.5 at the threshold, 1.0 when orthogonal."""
    distance = (threshold - best_sim) / max(1e-6, threshold + 1.0)
    return round(float(np.clip(0.5 + distance * 2.0, 0.5, 1.0)), 3)
