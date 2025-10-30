from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from . import config, utils


def get_duration_seconds(video_path: Path) -> float:
    meta = utils.ffprobe_json(video_path)
    try:
        return float(meta.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


def compute_frame_step(video_path: Path, target_frames: int = config.FRAME_COUNT) -> int:
    meta = utils.ffprobe_json(video_path)
    nb_frames = None
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            nb = s.get("nb_frames")
            if nb is not None:
                try:
                    nb_frames = int(nb)
                except Exception:
                    nb_frames = None
            break
    if not nb_frames:
        # fallback by duration * fps estimation
        try:
            r_num, r_den = meta["streams"][0]["r_frame_rate"].split("/")
            fps = float(r_num) / float(r_den)
        except Exception:
            fps = 30.0
        duration = get_duration_seconds(video_path)
        nb_frames = max(1, int(duration * fps))
    step = max(1, nb_frames // max(1, target_frames))
    return step


def sample_frames(video_path: Path, out_dir: Path, resize_width: int = config.RESIZE_WIDTH,
                  target_frames: int = config.FRAME_COUNT) -> Tuple[List[Path], Dict]:
    utils.safe_mkdir(out_dir)
    # Ensure required tools exist
    utils.require_binaries(["ffmpeg", "ffprobe"])
    step = compute_frame_step(video_path, target_frames)
    vf = f"select='not(mod(n,{step}))',scale={resize_width}:-1"
    pattern = str(out_dir / "frame_%04d.png")
    code, out, err = utils.run_cmd([
        "ffmpeg", "-y", "-i", str(video_path), "-vf", vf, "-vsync", "0", pattern
    ])
    if code != 0:
        raise RuntimeError(f"ffmpeg sampling failed: {err}")

    frames = sorted(out_dir.glob("frame_*.png"))
    # Ensure we do not exceed target count due to rounding
    if len(frames) > target_frames:
        frames = frames[:target_frames]
    if len(frames) == 0:
        raise RuntimeError("No frames extracted; check input file and ffmpeg codecs support.")

    duration = get_duration_seconds(video_path)
    return frames, {"duration_sec": duration, "frame_step": step, "frame_count": len(frames)}


def load_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return img


