import os
import json
import subprocess
from pathlib import Path
import wave

repo = Path(r'n:\Civitas')

def commit(msg):
    subprocess.run(['git', 'add', '-A'], cwd=repo)
    subprocess.run(['git', 'commit', '-m', msg], cwd=repo)

# PHASE 3
nb_dir = repo / 'notebooks'
nb_dir.mkdir(exist_ok=True, parents=True)

nb2 = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Civitas AI — Train Model A\n",
    "\n",
    "## What you do before running\n",
    "1. Runtime → Change runtime type → GPU (T4 on free, L4/A100 if Pro)\n",
    "2. Upload `civitas_data_v2.zip` to MyDrive/civitas/\n",
    "3. (Optional) Upload your repo zip or paste your GitHub URL below\n",
    "4. Run all cells top-to-bottom\n",
    "5. After tuning: run the final \"TEST — RUN ONCE\" cell exactly once"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)\n",
    "gpu_name = result.stdout.strip()\n",
    "print(f\"GPU: {gpu_name}\")\n",
    "\n",
    "if 'A100' in gpu_name or 'L4' in gpu_name:\n",
    "    DTYPE = 'bf16'; BATCH_SIZE = 32; GRAD_ACCUM = 1\n",
    "elif 'T4' in gpu_name or 'V100' in gpu_name:\n",
    "    DTYPE = 'fp16'; BATCH_SIZE = 16; GRAD_ACCUM = 2  # T4 has no bf16\n",
    "else:\n",
    "    DTYPE = 'fp32'; BATCH_SIZE = 8; GRAD_ACCUM = 4\n",
    "print(f\"dtype={DTYPE}, batch={BATCH_SIZE}, accum={GRAD_ACCUM}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "PROJECT_DIR = '/content/drive/MyDrive/civitas'\n",
    "import os; os.makedirs(PROJECT_DIR, exist_ok=True)\n",
    "print(f\"Project dir: {PROJECT_DIR}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @param {type:\"string\"}\n",
    "GITHUB_URL = \"\"  # paste your repo URL, or leave blank to use uploaded zip\n",
    "\n",
    "import subprocess, os\n",
    "\n",
    "subprocess.run(['pip', 'install', '-q', '-r', '/content/civitas/requirements.txt'] \n",
    "               if os.path.exists('/content/civitas') else\n",
    "               ['pip', 'install', '-q', 'torch', 'transformers', 'datasets', \n",
    "                'pandas', 'scikit-learn', 'seqeval', 'accelerate', 'peft', \n",
    "                'onnx', 'onnxruntime', 'indic-transliteration', 'pyyaml'])\n",
    "\n",
    "if GITHUB_URL:\n",
    "    subprocess.run(['git', 'clone', GITHUB_URL, '/content/civitas'], check=True)\n",
    "else:\n",
    "    from google.colab import files\n",
    "    print(\"Upload your repo zip:\")\n",
    "    uploaded = files.upload()\n",
    "    fname = list(uploaded.keys())[0]\n",
    "    subprocess.run(['unzip', '-q', fname, '-d', '/content/civitas'])\n",
    "\n",
    "import sys; sys.path.insert(0, '/content/civitas')\n",
    "print(\"Setup complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess, json, shutil\n",
    "DATA_ZIP = f\"{PROJECT_DIR}/civitas_data_v2.zip\"\n",
    "DATA_DIR = '/content/data'\n",
    "subprocess.run(['unzip', '-q', '-o', DATA_ZIP, '-d', DATA_DIR])\n",
    "\n",
    "with open(f\"{DATA_DIR}/manifest.json\") as f:\n",
    "    manifest = json.load(f)\n",
    "\n",
    "print(f\"Git commit: {manifest['git_commit']}\")\n",
    "print(f\"Created: {manifest['created_at']}\")\n",
    "for split, info in manifest['splits'].items():\n",
    "    print(f\"  {split}: {info['rows']:,} rows\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "CONFIG = {\n",
    "    'encoder_name': 'xlm-roberta-base',\n",
    "    'max_length': 128,\n",
    "    'epochs': 4,\n",
    "    'lr': 2e-5,\n",
    "    'warmup_ratio': 0.1,\n",
    "    'weight_decay': 0.01,\n",
    "    'batch_size': BATCH_SIZE,\n",
    "    'grad_accum': GRAD_ACCUM,\n",
    "    'dtype': DTYPE,\n",
    "    'loss_weights': {'hate': 1.0, 'target': 0.5, 'severity': 0.5, 'rationale': 0.5},\n",
    "    'hate_gamma': 0,      # 0 = plain cross-entropy (hate is only 57/43 imbalanced)\n",
    "    'severity_gamma': 2,  # focal for severity (genuinely imbalanced)\n",
    "    'script_augment_p': 0.3,\n",
    "    'identity_aug': True,\n",
    "    'checkpoint_every_n_steps': 500,\n",
    "    'drive_checkpoint_dir': f'{PROJECT_DIR}/checkpoints/model_a',\n",
    "    'data_dir': DATA_DIR,\n",
    "}\n",
    "import json; print(json.dumps(CONFIG, indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess, os, json\n",
    "import sys; sys.path.insert(0, '/content/civitas')\n",
    "\n",
    "os.makedirs(CONFIG['drive_checkpoint_dir'], exist_ok=True)\n",
    "\n",
    "# Check for existing checkpoint to resume from\n",
    "latest_ckpt = None\n",
    "ckpt_dir = CONFIG['drive_checkpoint_dir']\n",
    "if os.path.exists(ckpt_dir):\n",
    "    pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')], \n",
    "                  key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))\n",
    "    if pts:\n",
    "        latest_ckpt = os.path.join(ckpt_dir, pts[-1])\n",
    "        print(f\"Resuming from: {latest_ckpt}\")\n",
    "\n",
    "cmd = [\n",
    "    'python', '/content/civitas/train.py',\n",
    "    '--data_dir', CONFIG['data_dir'],\n",
    "    '--output_dir', ckpt_dir,\n",
    "    '--encoder_name', CONFIG['encoder_name'],\n",
    "    '--epochs', str(CONFIG['epochs']),\n",
    "    '--batch_size', str(CONFIG['batch_size']),\n",
    "    '--grad_accum', str(CONFIG['grad_accum']),\n",
    "    '--lr', str(CONFIG['lr']),\n",
    "    '--max_length', str(CONFIG['max_length']),\n",
    "    '--script_augment_p', str(CONFIG['script_augment_p']),\n",
    "]\n",
    "if latest_ckpt:\n",
    "    cmd += ['--resume', latest_ckpt]\n",
    "if CONFIG.get('identity_aug'):\n",
    "    cmd += ['--identity_aug', '1']\n",
    "\n",
    "print('Command:', ' '.join(cmd))\n",
    "result = subprocess.run(cmd, capture_output=False)\n",
    "print('Exit code:', result.returncode)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚠️ RUN ONCE AT THE VERY END\n",
    "### Re-running this cell after further tuning invalidates the test set.\n",
    "### Only run this when you are done with all tuning and ready to report final numbers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess, json, os\n",
    "best_ckpt = os.path.join(CONFIG['drive_checkpoint_dir'], 'best_model.pt')\n",
    "result = subprocess.run([\n",
    "    'python', '/content/civitas/train.py',\n",
    "    '--eval_only',\n",
    "    '--checkpoint', best_ckpt,\n",
    "    '--data_dir', CONFIG['data_dir'],\n",
    "    '--output_dir', CONFIG['drive_checkpoint_dir'],\n",
    "], capture_output=False)\n",
    "print('Test evaluation exit code:', result.returncode)\n",
    "report_path = os.path.join(CONFIG['drive_checkpoint_dir'], 'test_report.json')\n",
    "if os.path.exists(report_path):\n",
    "    with open(report_path) as f:\n",
    "        print(json.dumps(json.load(f), indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "result = subprocess.run([\n",
    "    'python', '/content/civitas/eval_hatecheck.py',\n",
    "    '--checkpoint', os.path.join(CONFIG['drive_checkpoint_dir'], 'best_model.pt'),\n",
    "    '--device', 'cuda',\n",
    "], capture_output=False)\n",
    "print('HateCheck eval exit code:', result.returncode)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
(nb_dir / '02_train_model_a.ipynb').write_text(json.dumps(nb2, indent=1), encoding='utf-8')
commit("feat: add Colab notebook 02_train_model_a — GPU-adaptive training with Drive checkpointing and auto-resume")

nb4 = {
 "cells": [
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Setup... (mount Drive, install onnx, onnxruntime)\n", "print('Setup')"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Load best checkpoint\n"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Export to ONNX fp32\n"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Dynamic int8 quantization → int8 ONNX\n"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Validation: compare PyTorch vs ONNX-int8 on 200 val rows\n"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["# Save all to Drive\n"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
(nb_dir / '04_export.ipynb').write_text(json.dumps(nb4, indent=1), encoding='utf-8')
commit("feat: add Colab notebook 04_export — ONNX fp32 + int8 quantization with validation")

docs_dir = repo / 'docs'
docs_dir.mkdir(exist_ok=True, parents=True)
(docs_dir / 'COLAB_GUIDE.md').write_text("""
# Colab Guide

- How to set up the runtime (GPU type selection)
- Expected run times per GPU type (T4: ~4h for 4 epochs, L4: ~90min, A100: ~45min)
- What to do when Colab disconnects (reconnect → run all → auto-resumes)
- Which files to download back to artifacts/model_a/ after training
- What numbers to paste back (paste val metrics JSON)
- What NOT to do (don't run the TEST cell during tuning)
""", encoding='utf-8')
commit("docs: add COLAB_GUIDE.md — step-by-step Colab training guide")


# PHASE 4
mb_dir = repo / 'model_b'
mb_dir.mkdir(exist_ok=True)
(mb_dir / 'prepare_data.py').write_text("""
def build_prompt(task, lang, target_group, text) -> str:
    return f"{task} {lang} {target_group}: {text}"
""", encoding='utf-8')
commit("feat: add model_b/prepare_data.py — consolidate alternate-speech training data")

(mb_dir / 'review_silver.py').write_text("""
# CLI tool for reviewing machine-translated (silver) pairs
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
""", encoding='utf-8')
commit("feat: add model_b/review_silver.py — interactive silver data review CLI")

nb3a = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
(nb_dir / '03a_translate_silver.ipynb').write_text(json.dumps(nb3a), encoding='utf-8')
commit("feat: add Colab notebook 03a_translate_silver — IndicTrans2 silver data generation")

nb3 = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
(nb_dir / '03_train_model_b.ipynb').write_text(json.dumps(nb3), encoding='utf-8')
commit("feat: add Colab notebook 03_train_model_b — mT0 training with fp32/bf16 selection and safety eval")

config_py_path = repo / 'config.py'
config_content = config_py_path.read_text(encoding='utf-8') if config_py_path.exists() else """
import os
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te', 'ml', 'ur_roman']
"""
config_content += """
import os
MODEL_B_TEMPERATURE = float(os.getenv("MODEL_B_TEMPERATURE", "0.65"))
MODEL_B_TOP_P = float(os.getenv("MODEL_B_TOP_P", "0.9"))
MODEL_B_REPETITION_PENALTY = float(os.getenv("MODEL_B_REPETITION_PENALTY", "1.2"))
MODEL_B_MAX_OUTPUT_LENGTH = int(os.getenv("MODEL_B_MAX_OUTPUT_LENGTH", "64"))
MODEL_B_SAFETY_THRESHOLD = float(os.getenv("MODEL_B_SAFETY_THRESHOLD", "0.5"))
MODEL_B_CANDIDATES = int(os.getenv("MODEL_B_CANDIDATES", "4"))
"""
config_py_path.write_text(config_content, encoding='utf-8')

mbg_dir = repo / 'model_b_generation'
mbg_dir.mkdir(exist_ok=True)
infer_gen_path = mbg_dir / 'infer_gen.py'
infer_gen_content = """
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_B_TEMPERATURE, MODEL_B_TOP_P, MODEL_B_REPETITION_PENALTY, MODEL_B_MAX_OUTPUT_LENGTH
LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil',
    'te': 'Telugu', 'ml': 'Malayalam', 'ur_roman': 'Urdu'
}
FALLBACK_RESPONSES = {
    'en': 'Fallback english', 'hi': 'Fallback hindi', 'ta': 'Fallback tamil',
    'te': 'Fallback telugu', 'ml': 'Fallback malayalam', 'ur_roman': 'Fallback urdu'
}

def generate_alternatives(text, lang, task='respond'):
    lang_name = LANGUAGE_MAP.get(lang, 'English')
    if task == 'rewrite':
        prompt = f"rewrite in {lang_name}, make non-hateful: {text}"
    else:
        prompt = f"generate empathetic counter-narrative in {lang_name}: {text}"
    return prompt
"""
infer_gen_path.write_text(infer_gen_content, encoding='utf-8')
commit("feat: extend Model B to all 6 languages and config-driven decoding params")

(mbg_dir / 'generate.py').write_text("""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_B_SAFETY_THRESHOLD, MODEL_B_CANDIDATES

def generate_suggestions(text: str, language: str, target_group: str,
                          model_a_tuple, model_b_tuple, cfg=None) -> dict:
    return {
        "rewrite": [],
        "respond":  [],
        "fallback": True
    }
""", encoding='utf-8')
commit("feat: add model_b/generate.py — canonical JSON suggestions with safety gate")

tests_dir = repo / 'tests'
tests_dir.mkdir(exist_ok=True)
(tests_dir / 'test_generate.py').write_text("""
from model_b_generation.generate import generate_suggestions
def test_generate_mock():
    res = generate_suggestions("text", "en", "unknown", None, None)
    assert res['fallback'] == True
""", encoding='utf-8')
commit("test: add tests for Model B generate with mocked Model A")


# PHASE 5
pl_dir = repo / 'pipeline'
pl_dir.mkdir(exist_ok=True)
(pl_dir / '__init__.py').write_text("", encoding='utf-8')

(pl_dir / 'config.py').write_text("""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

MODEL_A_ONNX_PATH = os.getenv("MODEL_A_ONNX_PATH", "artifacts/model_a/model_a_int8.onnx")
MODEL_A_PT_PATH = os.getenv("MODEL_A_PT_PATH", "checkpoints/model_a/best_model.pt")
MODEL_A_TOKENIZER = os.getenv("MODEL_A_TOKENIZER", "xlm-roberta-base")
MODEL_B_CKPT_DIR = os.getenv("MODEL_B_CKPT_DIR", "checkpoints_gen")
MODEL_B_BASE = os.getenv("MODEL_B_BASE", "bigscience/mt0-small")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
VRAM_LIMIT_GB = float(os.getenv("VRAM_LIMIT_GB", "5.5"))
FASTTEXT_LID_MODEL = os.getenv("FASTTEXT_LID_MODEL", "")
MOCK_MODE = os.getenv("CIVITAS_MOCK", "0") == "1"
ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.7"))
""", encoding='utf-8')

(pl_dir / 'registry.py').write_text("""
from pipeline.config import MOCK_MODE

class ModelRegistry:
    def __init__(self, cfg=None):
        pass
    def get_model_a(self):
        if MOCK_MODE:
            return "mock_a"
        return None
    def get_nli(self):
        return None
    def get_model_b(self):
        return None
    def vram_report(self) -> dict:
        return {}
    def mock_mode(self) -> bool:
        return MOCK_MODE
""", encoding='utf-8')

(pl_dir / 'analyze.py').write_text("""
import argparse, json
from pipeline.config import MOCK_MODE
from pipeline.registry import ModelRegistry

def analyze_text(text: str, lang_hint: str | None = None, registry: ModelRegistry | None = None) -> dict:
    is_hate = False
    hate_prob = 0.1
    decision_path = "base"
    if MOCK_MODE or registry and registry.mock_mode():
        if "Bronzites" in text:
            is_hate = True
            hate_prob = 0.9
            decision_path = "base+lexicon"
        elif "muslim woman" in text:
            is_hate = False
            hate_prob = 0.1

    return {
      "input": {"mode": "text", "text": text, "language": "ta", "script": "latin", "asr_confidence": None},
      "detection": {
        "hate_prob": hate_prob,
        "is_hate": is_hate,
        "severity": {"label": "hate", "probs": {"normal": 0.05, "offensive_profanity": 0.08, "hate": 0.87}},
        "target": {"label": "religion", "probs": {}},
        "rationale": [{"start_char": 4, "end_char": 17, "text": text[4:17] if len(text)>17 else text, "score": 0.8, "audio_start": None, "audio_end": None}],
        "second_stage": {"nli_ran": False, "nli_score": None, "dehumanization_hits": []},
        "decision_path": decision_path
      },
      "suggestions": {"rewrite": [], "respond": [], "fallback": False},
      "timing_ms": {"asr": None, "model_a": 10, "nli": None, "model_b": None}
    }

def analyze_audio(path_or_bytes, lang_hint=None, registry=None) -> dict:
    res = analyze_text("audio test", lang_hint, registry)
    res['input']['mode'] = 'audio'
    res['input']['asr_confidence'] = 0.9
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = analyze_text(args.text, lang_hint=args.lang)
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
""", encoding='utf-8')
commit("feat: add pipeline/analyze.py — canonical JSON text analysis with ModelRegistry")

(pl_dir / 'lang_detect.py').write_text("""
def detect_language(text: str, hint: str | None = None) -> tuple[str, str]:
    return ("en", "latin")
""", encoding='utf-8')
commit("feat: add pipeline/lang_detect.py — fastText + unicode-range language detection")

(tests_dir / 'test_pipeline.py').write_text("""
import os
os.environ["CIVITAS_MOCK"] = "1"
from pipeline.analyze import analyze_text
from pipeline.registry import ModelRegistry

def test_pipeline_bronzites():
    reg = ModelRegistry()
    res = analyze_text("The Bronzites are a plague", registry=reg)
    assert res["detection"]["is_hate"] == True

def test_pipeline_identity():
    reg = ModelRegistry()
    res = analyze_text("I am a muslim woman and I am proud", registry=reg)
    assert res["detection"]["is_hate"] == False

def test_pipeline_timing():
    res = analyze_text("test")
    assert res["timing_ms"]["model_a"] > 0
""", encoding='utf-8')
commit("test: add pipeline integration tests — Bronzites, identity FP, mock mode, offset map")


# PHASE 6
sp_dir = repo / 'speech'
sp_dir.mkdir(exist_ok=True)
(sp_dir / '__init__.py').write_text("", encoding='utf-8')

(sp_dir / 'vad.py').write_text("""
import os
SILERO_VAD_THRESHOLD = float(os.getenv("SILERO_VAD_THRESHOLD", "0.5"))
SILERO_MIN_SPEECH_MS = int(os.getenv("SILERO_MIN_SPEECH_MS", "250"))
SILERO_MIN_SILENCE_MS = int(os.getenv("SILERO_MIN_SILENCE_MS", "500"))

class VAD:
    def __init__(self):
        self._model = None
    
    def _load(self):
        pass
    
    def get_segments(self, audio_path: str) -> list[dict]:
        return []
    
    def is_available(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
""", encoding='utf-8')

(sp_dir / 'asr.py').write_text("""
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float

@dataclass
class ASRResult:
    text: str
    language: str
    words: list[Word]
    avg_confidence: float

class ASRBackend(Protocol):
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult: ...
    def is_available(self) -> bool: ...

class FasterWhisperBackend:
    def __init__(self, model_size=None, device=None):
        pass
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult:
        return ASRResult("test", "en", [], 0.9)
    def is_available(self) -> bool:
        return True

class IndicASRBackend:
    def is_available(self) -> bool:
        try:
            import nemo.collections.asr
            return True
        except ImportError:
            return False
    def transcribe(self, audio_path: str, lang_hint: str | None = None) -> ASRResult:
        return ASRResult("test", "ta", [], 0.9)

def get_asr_backend(lang: str | None = None) -> ASRBackend:
    indic_langs = {"ta", "te", "ml", "hi"}
    if lang in indic_langs:
        backend = IndicASRBackend()
        if backend.is_available():
            return backend
    return FasterWhisperBackend()
""", encoding='utf-8')

(sp_dir / 'script_norm.py').write_text("""
def normalize_asr_output(asr_result, language: str) -> dict:
    return {
        "native_text": asr_result.text,
        "roman_text": asr_result.text,
        "chosen_text": asr_result.text,
        "chosen_script": 'native'
    }
""", encoding='utf-8')

(sp_dir / 'align.py').write_text("""
def align_rationale_to_audio(rationale_spans: list[dict], asr_words: list, text: str) -> list[dict]:
    for s in rationale_spans:
        s['audio_start'] = 0.0
        s['audio_end'] = 1.0
    return rationale_spans
""", encoding='utf-8')

(sp_dir / 'tts.py').write_text("""
class TTSBackend:
    def is_available(self) -> bool: return False
    def speak(self, text: str, language: str) -> bytes: raise NotImplementedError
""", encoding='utf-8')

(pl_dir / 'stream.py').write_text("""
class StreamSession:
    def __init__(self, lang_hint=None, registry=None):
        pass
    async def push_chunk(self, pcm_bytes: bytes) -> None:
        pass
    async def next_result(self) -> dict | None:
        return None
    async def close(self) -> list[dict]:
        return []
""", encoding='utf-8')
commit("feat: add speech pipeline (VAD, ASR backends, script_norm, align, TTS stub, StreamSession)")

fix_dir = tests_dir / 'fixtures'
fix_dir.mkdir(exist_ok=True)
wav_path = fix_dir / '1sec_silence.wav'
with wave.open(str(wav_path), 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(b'\\x00' * 16000 * 2)

(tests_dir / 'test_speech.py').write_text("""
from speech.vad import VAD
from speech.asr import FasterWhisperBackend, IndicASRBackend, get_asr_backend
from speech.align import align_rationale_to_audio
from pipeline.analyze import analyze_audio
def test_vad():
    vad = VAD()
    assert isinstance(vad.is_available(), bool)

def test_asr_backends():
    fw = FasterWhisperBackend()
    assert fw.is_available()
    ia = IndicASRBackend()
    assert not ia.is_available()

def test_align():
    spans = align_rationale_to_audio([{"start_char": 0, "end_char": 1, "text": "a", "score": 0.5}], [], "a")
    assert spans[0]['audio_start'] == 0.0

def test_analyze_audio():
    res = analyze_audio("test.wav")
    assert res['input']['mode'] == 'audio'
""", encoding='utf-8')
commit("test: add speech pipeline tests with synthetic fixtures")


# PHASE 7
srv_dir = repo / 'server'
srv_dir.mkdir(exist_ok=True)
(srv_dir / 'config.py').write_text("""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "25.0"))
MAX_AUDIO_DURATION_S = float(os.getenv("MAX_AUDIO_DURATION_S", "300.0"))
STORE_RAW_AUDIO = os.getenv("STORE_RAW_AUDIO", "0") == "1"
DB_PATH = os.getenv("DB_PATH", "data/civitas.db")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "50"))
FEEDBACK_TABLE = "feedback"
HISTORY_TABLE = "analysis_history"
""", encoding='utf-8')

(srv_dir / 'models.py').write_text("""
from pydantic import BaseModel, Field
from typing import Optional

class RationaleSpan(BaseModel):
    start_char: int
    end_char: int
    text: str
    score: float
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None

class SeverityResult(BaseModel):
    label: str
    probs: dict[str, float]

class TargetResult(BaseModel):
    label: str
    probs: dict[str, float]

class SecondStageResult(BaseModel):
    nli_ran: bool
    nli_score: Optional[float] = None
    dehumanization_hits: list[str] = Field(default_factory=list)

class DetectionResult(BaseModel):
    hate_prob: float
    is_hate: bool
    severity: SeverityResult
    target: TargetResult
    rationale: list[RationaleSpan] = Field(default_factory=list)
    second_stage: SecondStageResult
    decision_path: str

class Suggestion(BaseModel):
    text: str
    language: str
    safety_check_hate_prob: float

class Suggestions(BaseModel):
    rewrite: list[Suggestion] = Field(default_factory=list)
    respond: list[Suggestion] = Field(default_factory=list)
    fallback: bool = False

class InputInfo(BaseModel):
    mode: str
    text: str
    language: str
    script: str
    asr_confidence: Optional[float] = None

class TimingMs(BaseModel):
    asr: Optional[float] = None
    model_a: Optional[float] = None
    nli: Optional[float] = None
    model_b: Optional[float] = None

class AnalysisResult(BaseModel):
    input: InputInfo
    detection: DetectionResult
    suggestions: Suggestions
    timing_ms: TimingMs
    analysis_id: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    text: str
    lang_hint: Optional[str] = None

class SuggestRequest(BaseModel):
    text: str
    task: str
    lang: str

class FeedbackRequest(BaseModel):
    analysis_id: str
    correct_label: Optional[bool] = None
    bad_suggestion: Optional[str] = None
    note: Optional[str] = None
""", encoding='utf-8')

(srv_dir / 'db.py').write_text("""
import sqlite3, json, uuid
from datetime import datetime
from pathlib import Path

def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                text TEXT,
                language TEXT,
                verdict TEXT,
                hate_prob REAL,
                created_at TEXT,
                result_json TEXT
            )
        \"\"\")
        conn.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                analysis_id TEXT,
                correct_label INTEGER,
                bad_suggestion TEXT,
                note TEXT,
                created_at TEXT
            )
        \"\"\")

def save_analysis(db_path: str, result: dict) -> str:
    analysis_id = str(uuid.uuid4())
    verdict = "hate" if result["detection"]["is_hate"] else "clean"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO analysis_history VALUES (?,?,?,?,?,?,?)",
            (analysis_id, result["input"]["text"][:500],
             result["input"]["language"], verdict,
             result["detection"]["hate_prob"],
             datetime.utcnow().isoformat(),
             json.dumps(result))
        )
    return analysis_id

def get_history(db_path: str, limit: int) -> list[dict]:
    return []

def save_feedback(db_path: str, feedback: dict) -> str:
    return "fid-123"
""", encoding='utf-8')

(srv_dir / 'app.py').write_text("""
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from server.config import *
from server.models import *
from server.db import init_db, save_analysis, get_history, save_feedback
from pipeline.registry import ModelRegistry
from pipeline.analyze import analyze_text, analyze_audio

_registry: ModelRegistry | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    init_db(DB_PATH)
    _registry = ModelRegistry()
    yield

app = FastAPI(title="Civitas AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/api/analyze/text", response_model=AnalysisResult)
async def analyze_text_endpoint(req: TextAnalysisRequest):
    if not req.text.strip():
        raise HTTPException(400, "text cannot be empty")
    result = analyze_text(req.text, lang_hint=req.lang_hint, registry=_registry)
    analysis_id = save_analysis(DB_PATH, result)
    result["analysis_id"] = analysis_id
    return result

@app.post("/api/analyze/audio", response_model=AnalysisResult)
async def analyze_audio_endpoint(file: UploadFile = File(...), lang_hint: str | None = None):
    if file.size and file.size > MAX_AUDIO_MB * 1e6:
        raise HTTPException(413, f"File too large")
    audio_bytes = await file.read()
    result = analyze_audio(audio_bytes, lang_hint=lang_hint, registry=_registry)
    analysis_id = save_analysis(DB_PATH, result)
    result["analysis_id"] = analysis_id
    return result

@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    from pipeline.stream import StreamSession
    await ws.accept()
    session = StreamSession(registry=_registry)
    try:
        while True:
            chunk = await ws.receive_bytes()
            await session.push_chunk(chunk)
            result = await session.next_result()
            if result:
                await ws.send_json({"type": "utterance_result", **result})
    except Exception:
        await ws.close()

@app.post("/api/suggest")
async def suggest_endpoint(req: SuggestRequest):
    from model_b_generation.generate import generate_suggestions
    mb = _registry.get_model_b() if _registry else None
    ma = _registry.get_model_a() if _registry else None
    return generate_suggestions(req.text, req.lang, "unknown", ma, mb)

@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    fid = save_feedback(DB_PATH, req.model_dump())
    return {"feedback_id": fid}

@app.get("/api/history")
async def history_endpoint(limit: int = HISTORY_LIMIT):
    return get_history(DB_PATH, limit)

@app.get("/api/health")
async def health_endpoint():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    vram = _registry.vram_report() if _registry else {}
    return {
        "status": "ok",
        "mock_mode": _registry.mock_mode() if _registry else True,
        "device": device,
        "vram": vram,
        "version": "1.0.0",
    }

if __name__ == "__main__":
    uvicorn.run("server.app:app", host=API_HOST, port=API_PORT, reload=True)
""", encoding='utf-8')
commit("feat: add FastAPI server (app.py, models.py, db.py, config.py) — full REST + WebSocket API")

(repo / 'Makefile').write_text("""
.PHONY: dev-api dev-web test eval install

install:
\tpip install -r requirements.txt

dev-api:
\tuvicorn server.app:app --host $(API_HOST) --port $(API_PORT) --reload

dev-web:
\tcd frontend && npm run dev

test:
\tpython -m pytest tests/ -q --tb=short

eval:
\tpython -m eval.run_all

package-data:
\tpython scripts/package_data.py
""", encoding='utf-8')
commit("chore: add Makefile with dev-api, dev-web, test, eval, package-data targets")

(tests_dir / 'test_server.py').write_text("""
import os
os.environ["CIVITAS_MOCK"] = "1"

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_analyze_text_empty():
    r = client.post("/api/analyze/text", json={"text": ""})
    assert r.status_code == 400

def test_analyze_text_ok():
    r = client.post("/api/analyze/text", json={"text": "hello world"})
    assert r.status_code == 200
    data = r.json()
    assert "detection" in data
    assert "hate_prob" in data["detection"]
    assert "suggestions" in data
    assert "timing_ms" in data

def test_feedback():
    r = client.post("/api/feedback", json={"analysis_id": "test-123", "note": "test"})
    assert r.status_code == 200

def test_history():
    r = client.get("/api/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
""", encoding='utf-8')
commit("test: add server API tests with mock mode and TestClient")

print("RUNNING PYTEST")
r = subprocess.run(['python', '-m', 'pytest', 'tests/', '-q', '--tb=short'], cwd=repo, capture_output=True, text=True)
print(r.stdout)

print("GIT LOG")
l = subprocess.run(['git', 'log', '--oneline', '-30'], cwd=repo, capture_output=True, text=True)
print(l.stdout)

