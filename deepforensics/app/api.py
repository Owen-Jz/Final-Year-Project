from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config, ingest, metadata as metadata_mod, ml as ml_mod, prnu as prnu_mod, ensemble as ensemble_mod, utils


config.ensure_dirs()
app = FastAPI(title=config.APP_NAME, version=config.VERSION)

# Serve UI at /ui to avoid intercepting API POSTs
ui_dir = config.BASE_DIR / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="static")


@app.get("/")
def root_redirect():
    if ui_dir.exists():
        return RedirectResponse(url="/ui/")
    return {"message": "DeepForensics API"}


def _refuse_external_calls_guard():
    # Placeholder for future: ensure no external endpoints are configured
    # Since this app makes no network calls by design, nothing to do here.
    return


@app.get("/health")
def health():
    return {"status": "ok", "version": config.VERSION}


@app.post("/enroll")
async def enroll(device_id: str = Form(...), files: List[UploadFile] | None = None):
    _refuse_external_calls_guard()
    if not files:
        raise HTTPException(400, "No files provided")

    tmpdir = utils.create_temp_dir("enroll")
    try:
        all_residuals: List[np.ndarray] = []
        for f in files:
            in_path = tmpdir / f.filename
            with open(in_path, "wb") as out:
                out.write(await f.read())
            frames, _ = ingest.sample_frames(in_path, tmpdir / "frames")
            _, residuals = prnu_mod.process_frames_for_prnu(frames)
            all_residuals.extend(residuals)
        fingerprint = prnu_mod.aggregate_residuals(all_residuals)
        fp_dir = config.WORK_DIR / "device_fingerprints"
        utils.safe_mkdir(fp_dir)
        np.save(fp_dir / f"device_{device_id}.npy", fingerprint)
        return {"device_id": device_id, "status": "enrolled"}
    finally:
        utils.cleanup_path(tmpdir)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), privacy_mode: bool = Form(default=config.PRIVACY_MODE_DEFAULT), device_id: str | None = Form(default=None)):
    _refuse_external_calls_guard()
    started_at = datetime.utcnow().isoformat() + "Z"
    task_id = utils.make_task_id()
    tmpdir = utils.create_temp_dir("analyze")
    evidence_dir = utils.safe_mkdir(config.EVIDENCE_DIR / task_id)
    try:
        in_path = tmpdir / file.filename
        with open(in_path, "wb") as out:
            out.write(await file.read())

        # Ingest frames
        frames_dir = tmpdir / "frames"
        try:
            frames, ingest_info = ingest.sample_frames(in_path, frames_dir)
        except Exception as ie:
            raise HTTPException(400, f"Frame sampling failed: {ie}")

        # PRNU
        try:
            clip_prnu, residuals = prnu_mod.process_frames_for_prnu(frames)
        except Exception as pe:
            raise HTTPException(500, f"PRNU processing failed: {pe}")
        face_scores, heatmaps = prnu_mod.face_region_scores_and_heatmaps(frames, residuals, evidence_dir)

        # Metadata
        try:
            meta_details, meta_flags, meta_score = metadata_mod.analyze(in_path)
        except Exception as me:
            raise HTTPException(500, f"Metadata analysis failed: {me}")

        # ML provider (stub by default, optional Ollama if enabled)
        face_scores_json = [
            {"frame_index": s.frame_index, "bbox": list(s.bbox), "score": s.score} for s in face_scores
        ]
        frame_b64 = None
        if config.ML_PROVIDER == "ollama" and config.OLLAMA_ENABLE_VISION:
            # Prepare a few frame thumbnails as base64
            frame_b64 = []
            import base64
            for p in frames[:3]:
                try:
                    with open(p, "rb") as fh:
                        frame_b64.append(base64.b64encode(fh.read()).decode())
                except Exception:
                    pass
        ml_out = ml_mod.predict(in_path, meta_flags, face_scores_json, frame_b64)

        # PRNU similarity: if device enrolled, compare to fingerprint; else use proxy from faces
        prnu_similarity = 0.0
        prnu_reference_used = False
        if device_id:
            fp_path = config.WORK_DIR / "device_fingerprints" / f"device_{device_id}.npy"
            if fp_path.exists():
                try:
                    ref = np.load(fp_path)
                    prnu_similarity = float(prnu_mod.correlation_similarity(clip_prnu, ref))
                    # map from [-1,1] to [0,1]
                    prnu_similarity = (prnu_similarity + 1.0) / 2.0
                    prnu_reference_used = True
                except Exception:
                    prnu_reference_used = False
        if not prnu_reference_used:
            if face_scores:
                prnu_similarity = 1.0 - max(s.score for s in face_scores)
                prnu_similarity = float(np.clip(prnu_similarity, 0.0, 1.0))

        ens = ensemble_mod.score_and_decide(ml_out["score"], prnu_similarity, meta_score)

        # Serialize outputs
        # Save a representative residual image
        residual_paths: List[str] = []
        if residuals:
            import cv2
            rep = (residuals[0] - residuals[0].min())
            if rep.max() > 0:
                rep = rep / rep.max()
            rep_img = (rep * 255).astype("uint8")
            rep_path = evidence_dir / "residual_sample.png"
            cv2.imwrite(str(rep_path), rep_img)
            residual_paths.append(str(rep_path))

        # Include up to 3 heatmaps; base64 when privacy_mode, else paths
        heatmap_repr = None
        heatmap_list: List[str] | None = None
        if heatmaps:
            heatmap_list = []
            for hp in heatmaps[:3]:
                if privacy_mode:
                    heatmap_list.append("data:image/png;base64," + utils.b64_of_file(hp))
                else:
                    heatmap_list.append(hp.as_posix())
            heatmap_repr = heatmap_list[0]

        report: Dict = {
            "task_id": task_id,
            "source": {
                "filename": Path(file.filename).name,
                "filesize": Path(in_path).stat().st_size,
                "duration_sec": ingest_info.get("duration_sec", 0.0),
            },
            "ml": ml_out,
            "metadata": {"flags": meta_flags, "details": meta_details},
            "prnu": {
                "clip_score": float(np.mean(np.abs(clip_prnu))),
                "similarity": prnu_similarity,
                "reference_used": prnu_reference_used,
                "face_region_scores": face_scores_json,
                "heatmap_image": heatmap_repr,
                "heatmap_images": heatmap_list,
                "residual_images": residual_paths,
            },
            "ensemble": {
                **ens,
                "explanation": "local_stub+PRNU proxy+metadata rules",
            },
            "evidence": {
                "frames": [p.as_posix() for p in frames[:3]],
                "residuals": residual_paths,
            },
            "timestamps": {"started_at": started_at, "finished_at": datetime.utcnow().isoformat() + "Z"},
        }

        # Save report
        utils.safe_mkdir(config.REPORTS_DIR)
        report_path = config.REPORTS_DIR / f"{task_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Privacy cleanup
        if privacy_mode:
            utils.cleanup_path(evidence_dir)
            utils.cleanup_path(tmpdir)

        return JSONResponse(report)
    except HTTPException:
        # cleanup then re-raise
        try:
            if privacy_mode:
                utils.cleanup_path(evidence_dir)
                utils.cleanup_path(tmpdir)
        finally:
            pass
        raise
    except Exception as e:
        # Ensure cleanup if privacy
        try:
            if privacy_mode:
                utils.cleanup_path(evidence_dir)
                utils.cleanup_path(tmpdir)
        finally:
            pass
        raise HTTPException(500, f"Analysis failed: {e}")


@app.get("/report/{task_id}")
def get_report(task_id: str):
    path = config.REPORTS_DIR / f"{task_id}.json"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


