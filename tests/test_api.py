from pathlib import Path
import io
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient

from deepforensics.app.api import app
from deepforensics.app import utils


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available")
def test_analyze_endpoint_returns_report():
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as td:
        video = Path(td) / "gen.mp4"
        code, out, err = utils.run_cmd([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=1", str(video)
        ])
        assert code == 0
        data = {"privacy_mode": "true"}
        with open(video, "rb") as fh:
            files = {"file": (video.name, fh, "video/mp4")}
            r = client.post("/analyze", files=files, data=data)
        assert r.status_code == 200, r.text
        j = r.json()
        for k in ["task_id", "source", "ml", "metadata", "prnu", "ensemble", "timestamps"]:
            assert k in j

