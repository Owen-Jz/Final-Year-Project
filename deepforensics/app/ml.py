from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from . import config
from . import utils
import requests


def stub_predict(video_path: Path, metadata_flags: list[str], face_region_scores: list[dict]) -> Dict:
    """
    Deterministic local stub with simple heuristics. No network calls.
    Optional filename rules in examples/stub_rules.json
    """
    base_score = 0.12
    rule = "baseline_low"

    if "recompression_detected" in metadata_flags:
        base_score = 0.70
        rule = "recompression_high"

    # Face manipulation heuristic: high suspiciousness among face scores
    if face_region_scores:
        max_face = max(s.get("score", 0.0) for s in face_region_scores)
        if max_face >= 0.8:
            base_score = max(base_score, 0.85)
            rule = "face_artifact_very_high"
        elif max_face >= 0.6:
            base_score = max(base_score, 0.75)
            rule = "face_artifact_high"

    # Optional per-filename overrides
    if config.STUB_RULES_PATH.exists():
        try:
            with open(config.STUB_RULES_PATH, "r", encoding="utf-8") as f:
                rules = json.load(f)
            name = Path(video_path).name
            if name in rules:
                base_score = float(rules[name])
                rule = "stub_rules_override"
        except Exception:
            pass

    return {
        "provider": "local_stub",
        "score": float(np.clip(base_score, 0.0, 1.0)),
        "raw_response": {"type": "stub", "rule": rule},
    }


def _ensure_local_host(url: str) -> None:
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        raise RuntimeError("Refusing non-local Ollama host. Set DF_OLLAMA_HOST to 127.0.0.1 only.")


def ollama_predict(
    video_path: Path,
    metadata_flags: list[str],
    face_region_scores: list[dict],
    frame_images_b64: list[str] | None = None,
) -> Dict:
    _ensure_local_host(config.OLLAMA_HOST)
    model = config.OLLAMA_MODEL
    prompt = (
        "You are a forensics assistant. Given metadata flags and PRNU face-region scores, "
        "return a scalar suspicion score in [0,1] where higher means more likely manipulated. "
        "Consider recompression, face-region inconsistencies, and any visual artifacts. "
        "Respond JSON: {\"score\": <float>, \"rationale\": <short>}"
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"metadata_flags: {metadata_flags}"},
        {"type": "text", "text": f"face_region_scores(max3): {face_region_scores[:3]}"},
    ]
    if config.OLLAMA_ENABLE_VISION and frame_images_b64:
        for b in frame_images_b64[:3]:
            content.append({"type": "image", "image": b})

    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": content}
        ],
        "stream": False,
    }
    try:
        r = requests.post(
            f"{config.OLLAMA_HOST}/v1/chat/completions",
            json=body,
            timeout=config.OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        # Try to find content text
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            j = json.loads(text)
            score = float(np.clip(float(j.get("score", 0.5)), 0.0, 1.0))
            rationale = j.get("rationale", "")
        except Exception:
            score = 0.5
            rationale = text[:200]
        return {
            "provider": f"ollama:{model}",
            "score": score,
            "raw_response": {"type": "ollama", "rationale": rationale},
        }
    except Exception as e:
        # Fallback to stub
        stub = stub_predict(video_path, metadata_flags, face_region_scores)
        stub["raw_response"]["ollama_error"] = str(e)
        return stub


def predict(video_path: Path, metadata_flags: list[str], face_region_scores: list[dict], frame_images_b64: list[str] | None = None) -> Dict:
    if config.ML_PROVIDER == "ollama":
        return ollama_predict(video_path, metadata_flags, face_region_scores, frame_images_b64)
    return stub_predict(video_path, metadata_flags, face_region_scores)


