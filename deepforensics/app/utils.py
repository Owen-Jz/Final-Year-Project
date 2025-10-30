from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config


def redact_path(path: str) -> str:
    try:
        p = Path(path)
        return f".../{p.name}"
    except Exception:
        return "<redacted>"


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")


def ffprobe_json(video_path: Path) -> Dict:
    code, out, err = run_cmd([
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return json.loads(out or "{}")


def exiftool_dict(video_path: Path) -> Optional[Dict]:
    # Optional; skip if not installed
    if shutil.which("exiftool") is None:
        return None
    code, out, err = run_cmd(["exiftool", str(video_path)])
    if code != 0:
        return None
    result: Dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def make_task_id() -> str:
    return str(uuid.uuid4())


def b64_of_file(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def safe_mkdir(path: Path) -> Path:
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                # Python <3.8 fallback
                if path.exists():
                    path.unlink()


def create_temp_dir(prefix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"{config.APP_NAME}_{prefix}_"))
    return d


def require_binaries(binaries: List[str]) -> None:
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        raise RuntimeError(
            f"Missing required binaries: {', '.join(missing)}. Install them and ensure they are on PATH."
        )


