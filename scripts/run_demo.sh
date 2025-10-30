#!/usr/bin/env bash
set -euo pipefail

python -m venv venv
if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == *"NT"* ]]; then
  source venv/Scripts/activate
else
  source venv/bin/activate
fi

pip install -U pip
pip install -r requirements.txt

echo "Starting server at http://localhost:8000"
uvicorn deepforensics.app.api:app --reload --port 8000


