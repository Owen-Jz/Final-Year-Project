# Gemma2 Setup Guide for DeepForensics

## Quick Setup

### 1. Install Gemma2 Model in Ollama

```bash
ollama pull gemma2:9b
```

Or for a smaller model:

```bash
ollama pull gemma2:2b
```

### 2. Start Ollama Service

**Terminal 1** (keep running):

```bash
ollama serve
```

### 3. Configure DeepForensics (Already Done!)

The project is now configured to use Gemma2 by default:

- `ML_PROVIDER = "ollama"` (in `deepforensics/app/config.py`)
- `OLLAMA_MODEL = "gemma2:9b"` (can be overridden via env var)

### 4. Run the Application

**Terminal 2**:

```bash
cd C:\Users\Owen Digitals\Downloads\Fintech Dashboard Design\deepforensics
venv\Scripts\activate
uvicorn deepforensics.app.api:app --port 8000
```

Or to override the model:

```bash
set DF_OLLAMA_MODEL=gemma2:2b
uvicorn deepforensics.app.api:app --port 8000
```

### 5. Use the UI

1. Open http://127.0.0.1:8000
2. Upload a video
3. Click "Analyze"
4. **View the "AI Explanation" section** to see Gemma2's detailed reasoning!

---

## What Gemma2 Provides

- **Detailed Explanation**: Multi-paragraph analysis explaining which signals raised concerns
- **Key Findings**: Bullet-point list of specific artifacts detected
- **Confidence Level**: Low/Medium/High confidence in the assessment
- **Score**: Manipulation likelihood from 0.0 (authentic) to 1.0 (manipulated)

All of this is displayed prominently in the UI and included in the JSON report.

---

## Troubleshooting

### If it still shows "ML: stub"

1. **Check Ollama is running**: `curl http://127.0.0.1:11434/api/tags`
2. **Verify model is pulled**: Should see `gemma2:9b` in the list
3. **Check env vars**: Ensure you're running uvicorn in the same terminal where env vars were set
4. **Check provider chip**: Top-right should show "ML: ollama:gemma2:9b"

### If explanation is missing

- Gemma2 might have returned free-form text instead of JSON. The system will try to extract the explanation anyway.
- Check the downloaded JSON report under `ml.raw_response.full_text` for the complete model response.

---

## Customization

To use a different Gemma variant or adjust timeout:

```bash
set DF_OLLAMA_MODEL=gemma2:2b    # Smaller, faster
set DF_OLLAMA_TIMEOUT=60         # Longer timeout for complex videos
```

Or edit `deepforensics/app/config.py` directly.

---

## Why Gemma2?

- **Text-focused**: Better at structured reasoning and explanation
- **Fast inference**: Quick response times even on CPU
- **Explainable**: Designed to provide clear, detailed justifications
- **Local-only**: All analysis happens on your machine

**Note**: Gemma2 is text-only, so it analyzes metadata flags and PRNU scores rather than visual frame content. For vision-based analysis, use `llava:7b` or similar vision-capable models (set `OLLAMA_ENABLE_VISION = True` in config).
