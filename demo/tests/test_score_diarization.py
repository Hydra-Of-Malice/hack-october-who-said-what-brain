from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import score_diarization as sd

TRUTH = [
    {"speaker": "Priya", "t_start": 0.5, "t_end": 5.0, "text": "Starter is 4,999 rupees with fifteen hundred shipments."},
    {"speaker": "Rahul", "t_start": 5.6, "t_end": 11.0, "text": "I'd cap the annual discount at fifteen percent."},
    {"speaker": "Meera", "t_start": 11.6, "t_end": 16.0, "text": "I'll send the deck to Saffron Basket by Thursday."},
    {"speaker": "Priya", "t_start": 16.5, "t_end": 21.0, "text": "We launch on November second."},
    {"speaker": "Arjun", "t_start": 21.7, "t_end": 27.0, "text": "The usage meter ships on Friday."},
]
TRACK_OF = {"Priya": "S3", "Rahul": "S1", "Meera": "S4", "Arjun": "S2"}  # deliberately scrambled


def utterances(rows: list[dict], track_of: dict[str, str] = TRACK_OF) -> list[dict]:
    return [{"speaker": track_of[r["speaker"]], "speaker_name": None, "t_start": r["t_start"] + 0.05,
             "t_end": r["t_end"] + 0.1, "text": r["text"]} for r in rows]


def test_perfect_listener_scores_full_marks_with_scrambled_tracks() -> None:
    result = sd.score(sd.load_truth(TRUTH), sd.load_hypothesis(utterances(TRUTH)))
    sp, tx = result["speakers"], result["text"]
    assert sp["mapping"] == {"S3": "Priya", "S1": "Rahul", "S4": "Meera", "S2": "Arjun"}
    assert sp["speaker_accuracy"] == pytest.approx(1.0)
    assert sp["turn_accuracy"] == pytest.approx(1.0)
    assert sp["utterance_accuracy"] == pytest.approx(1.0)
    assert sp["der"] < 0.01
    assert tx["wer"] == pytest.approx(0.0)
    assert tx["word_recall"] == pytest.approx(1.0)


def test_one_misattributed_turn_lowers_turn_and_time_accuracy() -> None:
    hyp = utterances(TRUTH)
    hyp[2]["speaker"] = TRACK_OF["Rahul"]  # Meera's turn credited to Rahul's track
    sp = sd.score(sd.load_truth(TRUTH), sd.load_hypothesis(hyp))["speakers"]
    assert sp["mapping"]["S1"] == "Rahul"
    assert sp["turn_correct"] == 4
    assert sp["speaker_accuracy"] < 0.9
    assert sp["confusion_s"] > 3.5


def test_split_speaker_leaves_extra_track_unmapped_and_counted_as_confusion() -> None:
    hyp = utterances(TRUTH)
    hyp[3]["speaker"] = "S9"  # Priya's second turn landed on a fresh track
    sp = sd.score(sd.load_truth(TRUTH), sd.load_hypothesis(hyp))["speakers"]
    assert "S9" not in sp["mapping"]
    assert sp["mapping"]["S3"] == "Priya"
    assert sp["turn_correct"] == 4
    assert 0.7 < sp["speaker_accuracy"] < 0.9


def test_merged_speakers_map_to_the_larger_one() -> None:
    track_of = dict(TRACK_OF, Arjun="S1")  # Arjun and Rahul merged into one track
    sp = sd.score(sd.load_truth(TRUTH), sd.load_hypothesis(utterances(TRUTH, track_of)))["speakers"]
    assert sp["mapping"]["S1"] in {"Rahul", "Arjun"}
    assert sp["turn_correct"] == 4


def test_ui_names_are_scored_case_insensitively() -> None:
    hyp = utterances(TRUTH)
    names = {"S3": "priya", "S1": "Rahul", "S4": "MEERA"}  # Arjun's track left unnamed
    for u in hyp:
        u["speaker_name"] = names.get(u["speaker"])
    named = sd.score(sd.load_truth(TRUTH), sd.load_hypothesis(hyp))["speakers"]["named"]
    assert named["named_tracks"] == 3
    assert 0.7 < named["accuracy"] < 1.0


def test_api_meeting_shape_and_bare_list_are_both_accepted() -> None:
    hyp = utterances(TRUTH)
    as_meeting = {"meeting": {"id": "m1"}, "speakers": [], "utterances": hyp, "decisions": []}
    assert sd.load_hypothesis(as_meeting) == sd.load_hypothesis(hyp)
    with pytest.raises(ValueError, match="t_start"):
        sd.load_hypothesis([{"speaker": "S1", "t_end": 2.0}])
    with pytest.raises(ValueError, match="no utterances"):
        sd.score(sd.load_truth(TRUTH), [])


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (2, 5), (5, 2), (4, 6), (6, 6)])
def test_assignment_matches_brute_force(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    for _ in range(25):
        overlap = rng.integers(0, 50, size=shape).astype(float)
        tracks = [f"S{i}" for i in range(shape[0])]
        names = [f"N{j}" for j in range(shape[1])]
        mapping = sd.best_mapping(overlap, tracks, names)
        got = sum(overlap[tracks.index(t), names.index(n)] for t, n in mapping.items())
        n_tracks, n_names = shape
        if n_tracks <= n_names:
            best = max(sum(overlap[r, cols[r]] for r in range(n_tracks))
                       for cols in itertools.permutations(range(n_names), n_tracks))
        else:
            best = max(sum(overlap[rows[c], c] for c in range(n_names))
                       for rows in itertools.permutations(range(n_tracks), n_names))
        assert got == pytest.approx(best)
        assert len(set(mapping.values())) == len(mapping)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Starter at 4,999 rupees a month", "Starter at ₹4999 a month"),
        ("fifteen hundred shipments", "1500 shipments"),
        ("twenty percent off", "20% off"),
        ("we launch on November second.", "We launch on November 2nd"),
        ("four thousand nine hundred and ninety-nine", "4,999"),
        ("It was 34 accounts.", "It was thirty-four accounts"),
        ("about 1.8 rupees, once", "about 1.8 rupees once"),
    ],
)
def test_text_normalisation_equates_spoken_and_written_forms(a: str, b: str) -> None:
    assert sd.normalise_text(a) == sd.normalise_text(b)


def test_wer_and_recall() -> None:
    ref = sd.normalise_text("Arjun ships the usage meter by Friday")
    assert sd.wer(ref, sd.normalise_text("Arjun ships the usage meter by Friday")) == 0.0
    assert sd.wer(ref, sd.normalise_text("Arjun ship the usage meter Friday")) == pytest.approx(2 / 7)
    assert sd.word_recall(ref, sd.normalise_text("meter usage the Friday")) == pytest.approx(4 / 7)


def test_cli_prints_report_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    truth = tmp_path / "t.truth.json"
    listener = tmp_path / "meeting.json"
    truth.write_text(json.dumps({"meeting_id": "demo", "turns": TRUTH}), encoding="utf-8")
    listener.write_text(json.dumps({"utterances": utterances(TRUTH)}), encoding="utf-8")
    assert sd.main([str(truth), str(listener)]) == 0
    assert "speaker acc    100.0 %" in capsys.readouterr().out
    assert sd.main([str(truth), str(listener), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["speakers"]["turn_correct"] == 5
    hyp = utterances(TRUTH)
    for u in hyp:
        u["speaker"] = "S1"
    listener.write_text(json.dumps(hyp), encoding="utf-8")
    assert sd.main([str(truth), str(listener), "--min-speaker-acc", "0.9"]) == 1
    assert sd.main([str(truth), str(tmp_path / "missing.json")]) == 2
