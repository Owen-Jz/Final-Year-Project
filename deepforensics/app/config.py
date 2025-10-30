from __future__ import annotations

import os
from pathlib import Path


# General
APP_NAME = "deepforensics"
VERSION = "0.1.0"

# Privacy
PRIVACY_MODE_DEFAULT = True
TMP_TTL_SECONDS = 3600

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORK_DIR = BASE_DIR / "work"
CACHE_DIR = WORK_DIR / "cache"
EVIDENCE_DIR = WORK_DIR / "evidence"
REPORTS_DIR = BASE_DIR / "examples" / "reports"
STUB_RULES_PATH = BASE_DIR / "examples" / "stub_rules.json"

# Ingest
FRAME_COUNT = 30
RESIZE_WIDTH = 640
MAX_WORKERS = max(1, min(4, os.cpu_count() or 2))

# PRNU
PRNU_FACE_CORR_SUSPICIOUS = 0.45
PRNU_FACE_CORR_LIKELY = 0.30

# Ensemble weights
W_ML = 0.6
W_PRNU = 0.3
W_META = 0.1

# ML provider
# "stub" (default) or "ollama" (local-only)
ML_PROVIDER = os.environ.get("DF_ML_PROVIDER", "stub")
OLLAMA_HOST = os.environ.get("DF_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("DF_OLLAMA_MODEL", "llava:7b")
OLLAMA_TIMEOUT = int(os.environ.get("DF_OLLAMA_TIMEOUT", "20"))
OLLAMA_ENABLE_VISION = True  # send a few frame thumbnails as base64 when available


def ensure_dirs() -> None:
    for p in [WORK_DIR, CACHE_DIR, EVIDENCE_DIR, REPORTS_DIR]:
        os.makedirs(p, exist_ok=True)


