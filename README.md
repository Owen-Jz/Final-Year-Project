# DeepForensics (Offline)

Privacy-first, fully offline forensic pipeline that analyzes a local video, computes PRNU + metadata signals, uses a local ML stub, and produces a JSON tamper report and heatmaps. No network calls. No Docker.

## Requirements

- Python 3.10+
- ffmpeg and ffprobe (installed on PATH)
- exiftool (optional; ffprobe fallback works)

Python packages are pinned in `requirements.txt`.

## Install ffmpeg/ffprobe and exiftool (commands only)

- Windows (choco):
  - `choco install ffmpeg`
  - `choco install exiftool`
- macOS (brew):
  - `brew install ffmpeg`
  - `brew install exiftool`
- Ubuntu/Debian:
  - `sudo apt update && sudo apt install -y ffmpeg exiftool`

## Local setup

From the repo root:

```bash
python -m venv venv
# Windows PowerShell
#   .\venv\Scripts\Activate.ps1
# Windows cmd
#   venv\Scripts\activate
# macOS/Linux
#   source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## Run the local demo (FastAPI + static UI)

```bash
uvicorn deepforensics.app.api:app --reload --port 8000
```

Then open `http://localhost:8000` and upload a local video. Default `privacy_mode=true` removes temporaries (heatmaps may not persist on disk when true; the UI tries to preview one inline if available).

Endpoints (local only):

- `POST /analyze` — multipart `file`, form field `privacy_mode` (default true). Returns full JSON report.
- `POST /enroll` — form field `device_id`, multiple `files[]` to build a device PRNU fingerprint (stored locally).
- `GET /report/{task_id}` — returns saved JSON report by id.
- `GET /health` — service status.

## Examples and evaluation

Place your local sample videos in `examples/` (not tracked). You may add a `examples/stub_rules.json` like:

```json
{
  "sample_fake_1.mp4": 0.85,
  "sample_pristine_1.mp4": 0.1
}
```

Open the notebook:

```bash
pip install jupyter
jupyter notebook evaluation/ensemble_evaluation.ipynb
```

Follow the notebook instructions to point to your local dataset. No downloads are performed automatically.

## Tests

Run all tests locally:

```bash
pytest -q
```

## Security & privacy

- Default `privacy_mode=true`: temporary frame folders and evidence are deleted after report generation.
- No external API calls. The app refuses to use any external endpoints by design.
- Logs avoid absolute paths where possible; file paths in reports are limited and can be removed with `privacy_mode=true`.

## Repo layout

```
deepforensics/
├─ app/
│  ├─ __init__.py
│  ├─ ingest.py
│  ├─ metadata.py
│  ├─ prnu.py
│  ├─ ml.py
│  ├─ ensemble.py
│  ├─ api.py
│  ├─ utils.py
│  └─ config.py
├─ ui/
│  ├─ index.html
│  └─ main.js
├─ tests/
│  ├─ test_ingest.py
│  ├─ test_prnu.py
│  ├─ test_metadata.py
│  └─ test_api.py
├─ evaluation/
│  └─ ensemble_evaluation.ipynb
├─ examples/
│  ├─ sample_pristine_1.mp4 (place locally)
│  ├─ sample_fake_1.mp4 (place locally)
│  └─ reports/
├─ requirements.txt
├─ README.md
└─ .github/workflows/ci.yml
```

## Helper script

`scripts/run_demo.sh` creates a venv, installs deps, and starts the server. See script content for details.
