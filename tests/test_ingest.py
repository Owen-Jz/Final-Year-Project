from pathlib import Path
import os
import tempfile
import shutil
import pytest

from deepforensics.app import ingest, utils


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available")
def test_sample_frames_returns_expected_count():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        video = td_path / "gen.mp4"
        # Generate a 2s 320x240, 10 fps test video
        code, out, err = utils.run_cmd([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2", str(video)
        ])
        assert code == 0, f"ffmpeg gen failed: {err}"
        frames, info = ingest.sample_frames(video, td_path / "frames", target_frames=10, resize_width=160)
        assert 1 <= len(frames) <= 10
        assert info["frame_count"] == len(frames)

