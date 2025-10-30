from pathlib import Path
import tempfile
import shutil
import pytest

from deepforensics.app import metadata as metadata_mod, utils


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available")
def test_metadata_analyze_structure():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        video = td_path / "gen.mp4"
        code, out, err = utils.run_cmd([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=1", str(video)
        ])
        assert code == 0
        details, flags, score = metadata_mod.analyze(video)
        assert isinstance(details, dict)
        assert isinstance(flags, list)
        assert 0.0 <= score <= 1.0

