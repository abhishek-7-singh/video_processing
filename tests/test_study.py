"""Unit tests for video study guide generation."""

import json
from pathlib import Path
from video_sentiment.study import study_video, _heuristic_study_guide, _format_study_markdown


def test_heuristic_study_guide():
    utterances = [
        {
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": 4.0,
            "text": "Product X has superior performance.",
            "targets": [{"surface": "Product X", "sentiment": "positive", "evidence": "superior performance"}],
        },
        {
            "speaker": "Speaker 2",
            "start": 4.5,
            "end": 8.0,
            "text": "Product Y is too expensive.",
            "targets": [{"surface": "Product Y", "sentiment": "negative", "evidence": "too expensive"}],
        },
    ]
    summary = {}
    guide = _heuristic_study_guide(utterances, summary, "Test Video")
    assert "Test Video" in guide["overview"]
    assert len(guide["core_concepts"]) >= 2
    assert len(guide["speaker_perspectives"]) == 2
    assert len(guide["high_yield_test_facts"]) >= 2

    md = _format_study_markdown(guide)
    assert "# Video Study Guide" in md
    assert "Product X" in md


def test_study_video_full(tmp_path: Path):
    doc = {
        "source": "demo.mp4",
        "duration_sec": 10.0,
        "speakers": ["Speaker 1"],
        "utterances": [
            {
                "speaker": "Speaker 1",
                "start": 0.0,
                "end": 5.0,
                "text": "The project deadline is Friday.",
                "targets": [{"surface": "deadline", "sentiment": "neutral", "evidence": "is Friday"}],
            }
        ],
        "summary": {"by_speaker": {"Speaker 1": {"deadline": {"dominant_label": "neutral"}}}},
    }
    result = study_video(doc, tmp_path)
    assert (tmp_path / "study_guide.json").exists()
    assert (tmp_path / "study_guide.md").exists()
    assert "overview" in result
