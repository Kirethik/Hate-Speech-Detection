# Civitas AI on a Hugging Face Docker Space (free CPU: 2 vCPU, 16 GB RAM).
# Build context = the folder scripts/make_space.py assembles (code + built UI + this file).
#
# Three Python environments, like the local setup, because the TTS engines pin conflicting deps:
#   /home/user/venv-api   API + ASR + classifiers   (requirements-voice.txt)
#   /home/user/venv-cb    Chatterbox TTS (en, hi)   (torch 2.6)
#   /home/user/venv-f5    IndicF5 TTS (ta, te, ml…) (transformers 4.x, numpy 1.x)
# constraints-*.txt are frozen from the environments verified on the dev laptop.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates zstd libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optional in-container LLM (used only when no external CIVITAS_LLM_BASE_URL secret is set)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface OLLAMA_MODELS=/home/user/.ollama/models
WORKDIR /home/user/app

ARG TORCH_CPU=https://download.pytorch.org/whl/cpu

# --- API environment
COPY --chown=user requirements-voice.txt constraints-api.txt ./
RUN python -m venv /home/user/venv-api \
    && /home/user/venv-api/bin/pip install torch torchaudio --index-url $TORCH_CPU \
    && /home/user/venv-api/bin/pip install -r requirements-voice.txt -c constraints-api.txt

# --- Chatterbox environment
COPY --chown=user tts_service/requirements-chatterbox.txt constraints-cb.txt ./
RUN python -m venv /home/user/venv-cb \
    && /home/user/venv-cb/bin/pip install "setuptools<81" torch==2.6.0 torchaudio==2.6.0 --index-url $TORCH_CPU \
    && /home/user/venv-cb/bin/pip install -r requirements-chatterbox.txt -c constraints-cb.txt

# --- IndicF5 environment
COPY --chown=user tts_service/requirements-indicf5.txt constraints-f5.txt ./
RUN python -m venv /home/user/venv-f5 \
    && /home/user/venv-f5/bin/pip install torch torchaudio --index-url $TORCH_CPU \
    && /home/user/venv-f5/bin/pip install -r requirements-indicf5.txt -c constraints-f5.txt

# --- Pre-download model weights into the image so restarts don't re-download ~10 GB.
# IndicF5 is gated: add HF_TOKEN as a Space secret (it is exposed to the build as a secret mount).
COPY --chown=user space_download.py ./
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" /home/user/venv-api/bin/python space_download.py

# --- In-container LLM weights (skipped at runtime if CIVITAS_LLM_BASE_URL points elsewhere)
ARG OLLAMA_MODEL=gemma4:e4b
RUN (ollama serve >/tmp/ollama-build.log 2>&1 &) \
    && for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done \
    && ollama pull $OLLAMA_MODEL

# --- App code + built web UI
COPY --chown=user . .
RUN chmod +x start.sh

ENV CIVITAS_STATIC_DIR=/home/user/app/web \
    CIVITAS_ENABLE_INDIC_ASR=false \
    CIVITAS_WARMUP=0 \
    CIVITAS_TTS_ROUTES=indicf5=http://127.0.0.1:9001,chatterbox=http://127.0.0.1:9002 \
    CORS_ORIGINS=*

EXPOSE 7860
CMD ["./start.sh"]
