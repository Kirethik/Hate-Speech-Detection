#!/usr/bin/env bash
# Space entrypoint. The API (port 7860) starts at once so the link works immediately;
# TTS engines and the optional local LLM start in the background ONE AT A TIME, because
# loading them together overshoots the 16 GB RAM limit.
#
# Space variables/secrets:
#   HF_TOKEN                 needed for IndicF5 (gated)
#   CIVITAS_LLM_BASE_URL     external OpenAI-compatible LLM (e.g. Groq); unset = Ollama in-container
#   CIVITAS_LLM_MODEL / CIVITAS_LLM_API_KEY
#   CIVITAS_SPACE_TTS        engines to run, default "chatterbox,indicf5"
set -u
cd /home/user/app

TTS="${CIVITAS_SPACE_TTS:-chatterbox,indicf5}"
LOCAL_LLM=0
if [ -z "${CIVITAS_LLM_BASE_URL:-}" ]; then
    LOCAL_LLM=1
    export CIVITAS_LLM_BASE_URL=http://127.0.0.1:11434/v1
    export CIVITAS_LLM_MODEL="${CIVITAS_LLM_MODEL:-gemma4:e4b}"
fi

wait_for() {  # url, seconds
    for _ in $(seq 1 "$2"); do curl -sf "$1" >/dev/null && return 0; sleep 1; done
    echo "[start] timed out waiting for $1"; return 1
}

start_tts() {  # engine, port, venv
    echo "[start] loading TTS $1 ..."
    CIVITAS_TTS_ENGINE="$1" "/home/user/$3/bin/python" -m uvicorn tts_service.app:app \
        --host 127.0.0.1 --port "$2" --log-level warning &
    wait_for "http://127.0.0.1:$2/health" 1800 && echo "[start] TTS $1 ready"
}

(
    case ",$TTS," in *,chatterbox,*) start_tts chatterbox 9002 venv-cb ;; esac
    case ",$TTS," in *,indicf5,*) start_tts indicf5 9001 venv-f5 ;; esac
    if [ "$LOCAL_LLM" = 1 ]; then
        echo "[start] starting Ollama ($CIVITAS_LLM_MODEL) ..."
        OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
            ollama serve >/tmp/ollama.log 2>&1 &
        wait_for http://127.0.0.1:11434/api/tags 120 && ollama pull "$CIVITAS_LLM_MODEL" \
            && echo "[start] LLM ready"
    fi
    echo "[start] background services done (ASR + classifiers load on the first request)"
) &

exec /home/user/venv-api/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
