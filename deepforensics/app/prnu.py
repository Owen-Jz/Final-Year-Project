from __future__ import annotations

import concurrent.futures as futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pywt

from . import config, ingest, utils


def _wavelet_denoise(gray: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec2(gray.astype(np.float32), "db2", level=2)
    cA, details = coeffs[0], coeffs[1:]
    # Soft-threshold detail coeffs
    new_details = []
    for (cH, cV, cD) in details:
        sigma = np.median(np.abs(cD)) / 0.6745 + 1e-6
        thr = sigma * np.sqrt(2 * np.log(gray.size))
        new_details.append((pywt.threshold(cH, thr, mode="soft"),
                            pywt.threshold(cV, thr, mode="soft"),
                            pywt.threshold(cD, thr, mode="soft")))
    denoised = pywt.waverec2([cA] + new_details, "db2")
    denoised = np.clip(denoised, 0, 255).astype(np.float32)
    return denoised


def extract_residual(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    den = _wavelet_denoise(gray)
    # Ensure same size due to possible wavelet reconstruction size drift
    if den.shape != gray.shape:
        den = cv2.resize(den, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    resid = gray - den
    resid -= resid.mean()
    std = resid.std() + 1e-6
    resid /= std
    return resid


def _residual_from_path(path: Path) -> np.ndarray:
    img = ingest.load_frame(path)
    return extract_residual(img)


def aggregate_residuals(residuals: List[np.ndarray]) -> np.ndarray:
    # Median aggregation
    stack = np.stack(residuals, axis=0)
    med = np.median(stack, axis=0)
    med -= med.mean()
    std = med.std() + 1e-6
    med /= std
    return med.astype(np.float32)


@dataclass
class FaceRegionScore:
    frame_index: int
    bbox: Tuple[int, int, int, int]
    score: float


def _detect_faces(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def create_heatmap_overlay(frame_bgr: np.ndarray, face_resid: np.ndarray, bg_resid: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    # Build a heat layer the same size as frame, fill with zeros, then place ROI heatmap
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    diff = np.abs(face_resid - bg_resid)
    diff = cv2.GaussianBlur(diff, (0, 0), 3)
    diff = diff - diff.min()
    if diff.max() > 0:
        diff = diff / diff.max()
    heat_roi = (diff * 255).astype(np.uint8)
    heat_color_roi = cv2.applyColorMap(heat_roi, cv2.COLORMAP_JET)

    heat_full = np.zeros_like(frame_bgr)
    # Clip bbox to frame bounds for safety
    x2, y2 = min(x + bw, w), min(y + bh, h)
    bw = max(0, x2 - x)
    bh = max(0, y2 - y)
    if bw > 0 and bh > 0 and heat_color_roi.shape[0] == bh and heat_color_roi.shape[1] == bw:
        heat_full[y:y+bh, x:x+bw] = heat_color_roi

    overlay = cv2.addWeighted(frame_bgr, 0.6, heat_full, 0.4, 0)
    return overlay


def process_frames_for_prnu(frames: List[Path]) -> Tuple[np.ndarray, List[np.ndarray]]:
    residuals: List[np.ndarray] = []
    with futures.ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        for resid in ex.map(_residual_from_path, frames):
            residuals.append(resid)
    clip_prnu = aggregate_residuals(residuals)
    return clip_prnu, residuals


def face_region_scores_and_heatmaps(frames: List[Path], residuals: List[np.ndarray], evidence_dir: Path) -> Tuple[List[FaceRegionScore], List[Path]]:
    scores: List[FaceRegionScore] = []
    heatmaps: List[Path] = []
    for idx, (frame_path, resid) in enumerate(zip(frames, residuals)):
        frame = ingest.load_frame(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces(gray)
        if not faces:
            continue
        # Background residual: blur to remove face detail
        bg_resid = cv2.GaussianBlur(resid, (31, 31), 0)
        for (x, y, w, h) in faces:
            face_r = resid[y:y+h, x:x+w]
            bg_r = bg_resid[y:y+h, x:x+w]
            if face_r.size == 0 or bg_r.size == 0:
                continue
            # Pearson correlation
            f = face_r.flatten()
            b = bg_r.flatten()
            if f.std() < 1e-6 or b.std() < 1e-6:
                corr = 1.0
            else:
                corr = float(np.corrcoef(f, b)[0, 1])
            score = 1.0 - max(-1.0, min(1.0, corr))  # lower corr -> higher suspiciousness
            scores.append(FaceRegionScore(frame_index=idx, bbox=(x, y, w, h), score=score))

            heat = create_heatmap_overlay(frame, face_r, bg_r, (x, y, w, h))
            out_path = evidence_dir / f"heatmap_frame_{idx:03d}.png"
            cv2.imwrite(str(out_path), heat)
            heatmaps.append(out_path)
    return scores, heatmaps


def correlation_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two residual maps, resizing b to a if needed."""
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)
    af = a.flatten()
    bf = b.flatten()
    if af.std() < 1e-8 or bf.std() < 1e-8:
        return 0.0
    corr = float(np.corrcoef(af, bf)[0, 1])
    return float(np.clip(corr, -1.0, 1.0))


