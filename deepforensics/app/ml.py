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
    explanation_parts = []

    if "recompression_detected" in metadata_flags:
        base_score = 0.70
        rule = "recompression_high"
        explanation_parts.append("Metadata analysis detected recompression artifacts, suggesting the video was re-encoded after initial recording.")

    # Face manipulation heuristic: high suspiciousness among face scores
    if face_region_scores:
        max_face = max(s.get("score", 0.0) for s in face_region_scores)
        avg_face = sum(s.get("score", 0.0) for s in face_region_scores) / len(face_region_scores)
        if max_face >= 0.8:
            base_score = max(base_score, 0.85)
            rule = "face_artifact_very_high"
            explanation_parts.append(f"PRNU analysis shows very high suspiciousness (score {max_face:.2f}) in face regions, indicating strong manipulation artifacts.")
        elif max_face >= 0.6:
            base_score = max(base_score, 0.75)
            rule = "face_artifact_high"
            explanation_parts.append(f"PRNU analysis detected elevated suspiciousness (score {max_face:.2f}) in face regions, suggesting possible manipulation.")
        elif avg_face > 0.5:
            explanation_parts.append(f"PRNU analysis shows moderate suspiciousness (average {avg_face:.2f}) across face regions.")
        else:
            explanation_parts.append("PRNU analysis shows low suspiciousness in face regions, indicating likely authentic content.")

    if not explanation_parts:
        explanation_parts.append("Basic analysis detected no significant manipulation indicators. For detailed frame-by-frame analysis, enable Ollama model.")

    # Optional per-filename overrides
    if config.STUB_RULES_PATH.exists():
        try:
            with open(config.STUB_RULES_PATH, "r", encoding="utf-8") as f:
                rules = json.load(f)
            name = Path(video_path).name
            if name in rules:
                base_score = float(rules[name])
                rule = "stub_rules_override"
                explanation_parts.append(f"Score overridden by stub_rules.json configuration.")
        except Exception:
            pass

    verdict = "AUTHENTIC" if base_score < 0.4 else ("MANIPULATED" if base_score >= 0.7 else "UNCERTAIN")
    
    return {
        "provider": "local_stub",
        "score": float(np.clip(base_score, 0.0, 1.0)),
        "raw_response": {
            "type": "stub",
            "rule": rule,
            "verdict": verdict,
            "explanation": " ".join(explanation_parts),
            "confidence": "low",
            "note": "Using local stub heuristics. Enable Ollama (set DF_ML_PROVIDER=ollama) for AI-powered frame-by-frame analysis.",
        },
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
    
    # Enhanced prompt for frame-by-frame explainable analysis
    prompt = """You are an expert video forensics analyst. Analyze the provided video frames visually along with forensic signals to determine if the video is AUTHENTIC or MANIPULATED (deepfake/face-swap/edited).

Forensic Signals:
- Metadata flags: Indicators of recompression, missing EXIF, timestamp inconsistencies
- Face-region PRNU scores: Low correlation between face and background regions suggests manipulation
  - Scores closer to 1.0 indicate high suspicion
  - Scores below 0.5 suggest likely authentic
  - Scores 0.5-0.7 are moderate/ambiguous

Your Task (CRITICAL - Perform BOTH):
1. FRAME-BY-FRAME ANALYSIS: Examine each frame image provided. For each frame, identify:
   - Visual artifacts (blur, inconsistent lighting, misaligned face boundaries, color mismatches)
   - Face realism (natural skin texture, proper blending with background)
   - Temporal consistency with other frames (if provided)
   - Any signs of manipulation specific to that frame

2. OVERALL ASSESSMENT: Synthesize frame analysis + forensic signals into final verdict:
   - Overall manipulation likelihood: 0.0 (definitely AUTHENTIC) to 1.0 (definitely MANIPULATED)
   - Which frames showed the strongest manipulation signs
   - Which signals (metadata, PRNU, visual) contributed most to the decision
   - Overall confidence level

Respond in valid JSON format:
{
  "score": <float between 0.0 and 1.0>,
  "verdict": "<AUTHENTIC | MANIPULATED | UNCERTAIN>",
  "explanation": "<detailed paragraph explaining overall reasoning, citing specific frames and signals>",
  "frame_analysis": [
    {
      "frame_index": 0,
      "frame_label": "first frame",
      "authentic": <true|false>,
      "visual_artifacts": ["<artifact 1>", "<artifact 2>", ...],
      "assessment": "<1-2 sentence explanation of why this frame appears authentic or manipulated>"
    },
    ...
  ],
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "confidence": "<low|medium|high>"
}"""
    
    # Build context with statistics
    max_face_score = max([s.get("score", 0.0) for s in face_region_scores], default=0.0) if face_region_scores else 0.0
    avg_face_score = sum([s.get("score", 0.0) for s in face_region_scores]) / len(face_region_scores) if face_region_scores else 0.0
    face_count = len(face_region_scores)
    
    context = f"""Video Analysis Context:
- Metadata flags: {metadata_flags if metadata_flags else 'None detected'}
- Face region analysis: {face_count} face regions detected
- Maximum face-region suspiciousness score: {max_face_score:.3f}
- Average face-region suspiciousness score: {avg_face_score:.3f}
- Top 3 face-region scores: {[f"{s.get('score', 0):.3f} (frame {s.get('frame_index', '?')})" for s in face_region_scores[:3]]}

IMPORTANT: You will receive {len(frame_images_b64) if frame_images_b64 else 0} frame images. Analyze each frame visually for manipulation artifacts, then provide:
1. Per-frame analysis (one entry per frame in frame_analysis array)
2. Overall verdict and explanation synthesizing all frames

Frame images will follow this message. Label them as "frame 0", "frame 1", "frame 2", etc. in your frame_analysis."""
    
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": context},
    ]
    if config.OLLAMA_ENABLE_VISION and frame_images_b64:
        for idx, b in enumerate(frame_images_b64[:3]):
            content.append({"type": "image", "image": b})
            # Add frame label hint
            content.append({"type": "text", "text": f"This is frame {idx}. Analyze it for visual manipulation artifacts."})

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
        # Extract content text from response
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        # Try to parse JSON response
        score = 0.5
        explanation = ""
        verdict = ""
        frame_analysis = []
        key_findings = []
        confidence = "medium"
        
        try:
            # Clean text in case it's wrapped in markdown code blocks
            cleaned = text.strip()
            if cleaned.startswith("```"):
                # Extract JSON from code block
                lines = cleaned.split("\n")
                start = next((i for i, l in enumerate(lines) if "{" in l), 0)
                end = next((i for i in range(len(lines)-1, -1, -1) if "}" in lines[i]), len(lines))
                cleaned = "\n".join(lines[start:end+1])
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            
            j = json.loads(cleaned)
            score = float(np.clip(float(j.get("score", 0.5)), 0.0, 1.0))
            explanation = j.get("explanation", j.get("rationale", ""))
            verdict = j.get("verdict", "")
            frame_analysis = j.get("frame_analysis", [])
            if not isinstance(frame_analysis, list):
                frame_analysis = []
            key_findings = j.get("key_findings", [])
            if isinstance(key_findings, str):
                key_findings = [key_findings]
            confidence = j.get("confidence", "medium")
        except Exception as parse_err:
            # Fallback: try to extract score and explanation from free text
            import re
            score_match = re.search(r'"score":\s*([\d.]+)', text)
            if score_match:
                score = float(np.clip(float(score_match.group(1)), 0.0, 1.0))
            explanation = text[:500] if len(text) > 500 else text
        
        return {
            "provider": f"ollama:{model}",
            "score": score,
            "raw_response": {
                "type": "ollama",
                "verdict": verdict,
                "explanation": explanation,
                "frame_analysis": frame_analysis if frame_analysis else None,
                "key_findings": key_findings if key_findings else None,
                "confidence": confidence,
                "full_text": text[:1000] if len(text) > 1000 else text,  # Keep truncated raw for debugging
            },
        }
    except Exception as e:
        # Fallback to stub
        stub = stub_predict(video_path, metadata_flags, face_region_scores)
        stub["raw_response"]["ollama_error"] = str(e)
        return stub


def predict(video_path: Path, metadata_flags: list[str], face_region_scores: list[dict], frame_images_b64: list[str] | None = None) -> Dict:
    if config.ML_PROVIDER == "ollama":
        try:
            return ollama_predict(video_path, metadata_flags, face_region_scores, frame_images_b64)
        except Exception as e:
            # Fallback to stub with error message
            stub_result = stub_predict(video_path, metadata_flags, face_region_scores)
            stub_result["raw_response"]["ollama_attempted"] = True
            stub_result["raw_response"]["ollama_error"] = str(e)
            stub_result["raw_response"]["note"] = f"Ollama was requested but failed: {str(e)}. Showing stub analysis instead. Ensure Ollama is running: 'ollama serve' and model is pulled: 'ollama pull {config.OLLAMA_MODEL}'"
            return stub_result
    
    return stub_predict(video_path, metadata_flags, face_region_scores)


