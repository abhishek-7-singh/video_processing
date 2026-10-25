"""Unit tests for screen capture and recording."""

from pathlib import Path
from video_sentiment.screen import capture_screen_snapshot, record_screen


def test_capture_screen_snapshot(tmp_path: Path):
    target = tmp_path / "screen.png"
    result = capture_screen_snapshot(output_path=target)
    assert result == target
    assert target.exists()
    assert target.stat().st_size > 0


def test_record_screen_short(tmp_path: Path):
    target = tmp_path / "test_rec.mp4"
    result = record_screen(target, duration_sec=1.0, fps=10)
    assert result.exists()
    assert result.stat().st_size > 0
