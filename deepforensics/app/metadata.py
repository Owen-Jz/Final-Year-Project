from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import utils


def analyze(video_path: Path) -> Tuple[Dict, List[str], float]:
    """
    Returns (details, flags, metadata_flag_score)
    """
    probe = utils.ffprobe_json(video_path)
    exif = utils.exiftool_dict(video_path)

    flags: List[str] = []
    details: Dict = {
        "create_time": None,
        "encoder": None,
        "recompression_chain": None,
    }

    # Gather some fields
    fmt = probe.get("format", {})
    tags = fmt.get("tags", {}) or {}
    create_time = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
    details["create_time"] = create_time

    encoder = fmt.get("encoder") or tags.get("encoder")
    details["encoder"] = encoder

    camera_model = None
    if exif:
        camera_model = exif.get("Model") or exif.get("Make")
        if exif.get("Create Date") and not create_time:
            details["create_time"] = exif.get("Create Date")

    if not camera_model:
        flags.append("missing_exif")

    # Simple recompression heuristic
    # If encoder string looks like software (e.g., Lavf, HandBrake), flag recompression
    enc_lower = (encoder or "").lower()
    if any(k in enc_lower for k in ["lavf", "handbrake", "ffmpeg", "premiere", "resolve", "x264", "x265"]):
        flags.append("recompression_detected")
        details["recompression_chain"] = encoder

    # Timestamp inconsistency: if creation_time missing but modification (from OS) exists, we cannot see here.
    # As a proxy, if streams have different start_time significantly
    try:
        streams = probe.get("streams", [])
        times = [float(s.get("start_time", 0.0)) for s in streams if s.get("start_time") is not None]
        if times and (max(times) - min(times)) > 5.0:
            flags.append("timestamp_inconsistency")
    except Exception:
        pass

    # Score in 0..1
    flag_weights = {
        "missing_exif": 0.3,
        "recompression_detected": 0.6,
        "timestamp_inconsistency": 0.3,
    }
    score = sum(flag_weights.get(f, 0.0) for f in flags)
    score = max(0.0, min(1.0, score))

    return details, flags, score


