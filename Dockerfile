# syntax=docker/dockerfile:1.7
# One image: the FastAPI service + the built web UI + the small listener models.
# Gemma 4 comes from the Gemini API (GEMINI_API_KEY) or an Ollama host (WSW_OLLAMA_URL).

# ---- web build ----
FROM node:22-slim AS web
WORKDIR /src
COPY brand.json ./brand.json
COPY apps/web/package.json apps/web/package-lock.json ./apps/web/
RUN cd apps/web && npm ci
COPY apps/web ./apps/web
RUN cd apps/web && npm run build

# ---- api ----
FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy \
    HF_HOME=/opt/hf-cache WSW_MODELS_DIR=/app/models WSW_DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/api/pyproject.toml apps/api/uv.lock apps/api/.python-version ./apps/api/
RUN cd apps/api && uv sync --frozen --no-dev --no-install-project
COPY brand.json ./brand.json
COPY apps/api ./apps/api
RUN cd apps/api && uv sync --frozen --no-dev
COPY scripts ./scripts
COPY demo ./demo
# Small listener models (VAD + speaker embeddings) and the ASR model, baked in so cold starts are fast.
ARG WSW_ASR_MODEL=small
ENV WSW_ASR_MODEL=${WSW_ASR_MODEL}
RUN /opt/venv/bin/python scripts/fetch_models.py \
 && /opt/venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('${WSW_ASR_MODEL}', device='cpu', compute_type='int8')"
# EmbeddingGemma needs a Hugging Face token if the repo is gated for you: pass --secret id=hf_token,src=...
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) /opt/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('onnx-community/embeddinggemma-300m-ONNX', allow_patterns=['onnx/model_quantized.onnx*','tokenizer*','*.json'])" \
 || echo "EmbeddingGemma not downloaded; the app falls back to the hash embedder (set WSW_EMBEDDER=hash)"
COPY --from=web /src/apps/web/dist ./apps/web/dist
EXPOSE 8000
WORKDIR /app/apps/api
CMD ["/opt/venv/bin/python", "-m", "wsw_brain", "--host", "0.0.0.0", "--port", "8000"]
