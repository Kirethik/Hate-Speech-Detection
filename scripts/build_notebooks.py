"""
Generates the Colab notebooks from the cell text below, so notebook code is
reviewed and diffed like any other code. Edit here, then run:

    python scripts/build_notebooks.py

Two self-contained notebooks, one per bundle made by scripts/make_bundles.py:
    notebooks/train_model_a.ipynb  (inside civitas_model_a/)
    notebooks/train_model_b.ipynb  (inside civitas_model_b/)
Each finds its bundle folder on Google Drive, trains with resume-after-disconnect,
and writes runs/<run>/results/ + civitas_results_<run>.zip next to the bundle.
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def write(name, cells):
    nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", NB_DIR / name)


def setup(bundle):
    return code(f'''
# Find the bundle on Google Drive, copy code + data to the fast local disk.
# Checkpoints and results go to <bundle>/runs/ on Drive, so they survive a disconnect.
from google.colab import drive
drive.mount("/content/drive")
import glob, hashlib, json, os, shutil, sys, zipfile
NAME = "{bundle}"
ROOTS = ["/content/drive/MyDrive", "/content/drive/MyDrive/civitas"]

def find_bundle():
    for r in ROOTS:
        for c in sorted(glob.glob(f"{{r}}/{{NAME}}*/code")) + sorted(glob.glob(f"{{r}}/{{NAME}}*/{{NAME}}/code")):
            return os.path.dirname(c)
    for r in ROOTS:  # only the zip was uploaded: unpack it next to itself
        z = f"{{r}}/{{NAME}}.zip"
        if os.path.exists(z):
            zipfile.ZipFile(z).extractall(r)
            return f"{{r}}/{{NAME}}"
    raise FileNotFoundError(f"Upload the {{NAME}} folder (or {{NAME}}.zip) to My Drive, then re-run.")

BUNDLE = find_bundle()
WORK, DATA, RUNS = "/content/work", "/content/data", f"{{BUNDLE}}/runs"
shutil.copytree(f"{{BUNDLE}}/code", WORK, dirs_exist_ok=True)
shutil.copytree(f"{{BUNDLE}}/data", DATA, dirs_exist_ok=True)
os.makedirs(RUNS, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

manifest = json.load(open(f"{{DATA}}/manifest.json"))
for name, digest in manifest["files"].items():
    h = hashlib.sha256(open(f"{{DATA}}/{{name}}", "rb").read()).hexdigest()
    assert h == digest, f"{{name}} is corrupted (sha256 mismatch) - re-upload the bundle"
print("bundle:", BUNDLE)
for split, info in manifest["splits"].items():
    print(f"  {{split:>6}}: {{info['rows']:>7}} rows  {{info.get('language_counts')}}")
print("built from commit", manifest["git_commit"], "at", manifest["created_at"])
''')


GPU_A = code('''
# GPU check: picks batch size / precision for this GPU
import subprocess, torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then Run all again."
GPU = torch.cuda.get_device_name(0)
major, _ = torch.cuda.get_device_capability(0)
BATCH, ACCUM = (64, 1) if "A100" in GPU else (32, 1)
print(f"{GPU}: batch {BATCH} x accum {ACCUM},", "bf16" if major >= 8 else "fp16 + GradScaler")
''')

# --------------------------------------------------------------------------- Model A
NB_A = [
    md('''
# Civitas — train Model A (hate speech detector)

XLM-RoBERTa encoder with 4 heads: hate yes/no, target group, severity, and which words are
hateful (rationale). Languages: en, hi, ta, te, ml, kn, Roman Urdu.

## WHAT YOU DO
1. Unzip `civitas_model_a.zip` on your PC and upload the **`civitas_model_a`** folder to
   **My Drive** (or upload the zip itself to My Drive; the notebook unpacks it).
2. Open `civitas_model_a/train_model_a.ipynb` from Drive with Google Colab.
3. **Runtime → Change runtime type → T4 GPU** (L4/A100 if you have Pro), then **Runtime → Run all**.
   Allow Drive access when asked.
4. If Colab disconnects: reconnect and **Run all** again. Training resumes from the last save
   (at most ~500 steps are repeated).
5. When it finishes, download **`civitas_model_a/runs/model_a/civitas_results_model_a.zip`** and
   **`best_model.pt`** from the same folder. Give both to Claude (put them in `artifacts/` on your PC:
   the zip in `artifacts/`, the .pt at `artifacts/model_a/best_model.pt`).
6. Do **not** tick CONFIRM in the last cell ("RUN ONCE") until Claude says tuning is finished.

Rough time on a free T4: 40–60 min per epoch; 4 epochs with early stopping ≈ 3–4 h,
across one or two sessions.
'''),
    GPU_A,
    setup("civitas_model_a"),
    code('''
# Install what training needs (Colab already has torch)
!pip -q install "transformers>=4.40" "indic-transliteration>=2.3.45" pyyaml pyarrow scikit-learn sentencepiece
import transformers
print("transformers", transformers.__version__)
'''),
    code('''
# Training config. Every key is a train.py flag (python train.py --help lists them all)
RUN_NAME = "model_a"  #@param {type:"string"}
OUT = f"{RUNS}/{RUN_NAME}"
CONFIG = {
    "data_dir": DATA,
    "output_dir": OUT,
    "encoder_name": "xlm-roberta-base",
    "epochs": 4,
    "batch_size": BATCH,
    "grad_accum": ACCUM,
    "lr": 2e-5,
    "max_length": 128,
    "warmup_ratio": 0.06,
    "evals_per_epoch": 3,
    "patience": 4,
    "save_every": 500,
    # loss
    "hate_gamma": 0.0,
    "severity_gamma": 2.0,
    "w_hate": 1.0, "w_target": 0.5, "w_severity": 0.5, "w_rationale": 0.5,
    # augmentation (train split only)
    "identity_aug": 1,
    "script_augment_p": 0.3,
    "spelling_augment_p": 0.15,
    "lang_weights": f"{DATA}/data_config.yaml",
}
ARGS = " ".join(f"--{k} {v!r}" if isinstance(v, str) else f"--{k} {v}" for k, v in CONFIG.items())
print(ARGS)
'''),
    code('''
# Train. Safe to re-run after a disconnect: --resume continues from OUT/last.pt
!cd {WORK} && python train.py {ARGS} --resume
'''),
    md('''
## Results (saved to files)
Scores the best checkpoint on **val** (never test), runs the robustness probe, then writes
`runs/model_a/results/` and `runs/model_a/civitas_results_model_a.zip`.
'''),
    code('''
m = json.load(open(f"{OUT}/best_metrics.json"))
keep = ["threshold", "hate_macro_f1", "hate_roc_auc", "severity_macro_f1", "target_macro_f1",
        "rationale_token_f1", "epoch", "global_step"]
print(json.dumps({k: m.get(k) for k in keep}, indent=1))
for s in ("hate_by_language", "hate_by_source"):
    print(s, {k: (round(v["macro_f1"], 3) if v else None) for k, v in m.get(s, {}).items()})
!cd {WORK} && python -c "import pandas as pd; pd.read_parquet('{DATA}/val.parquet').to_csv('/content/val.csv', index=False)"
!cd {WORK} && python -m eval.probe_model --checkpoint {OUT}/best_model.pt --eval_csv /content/val.csv --out {OUT}/robustness.json
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {OUT}/civitas_results_model_a.zip
print("download:", f"{OUT}/civitas_results_model_a.zip", "and", f"{OUT}/best_model.pt")
'''),
    md('''
---
# RUN ONCE — final report
**Only after Claude confirms tuning is finished.** The test split is the one honest number you
get; changing anything after looking at it makes it optimistic.
'''),
    code('''
CONFIRM = False  #@param {type:"boolean"}
assert CONFIRM, "Tick CONFIRM only when tuning is finished (see the note above)."
!cd {WORK} && python train.py --eval_only --split test --data_dir {DATA} --output_dir {OUT} --batch_size {BATCH}
!cd {WORK} && python -m eval.eval_hatecheck --checkpoint {OUT}/best_model.pt --out {OUT}/hatecheck.json
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {OUT}/civitas_results_model_a.zip
print("download again:", f"{OUT}/civitas_results_model_a.zip")
'''),
]

# --------------------------------------------------------------------------- Model B
NB_B = [
    md('''
# Civitas — train Model B (alternate speech generator)

`bigscience/mt0-base` + LoRA. Learns two tasks:
- **Rewrite**: say the same thing without the hate (used in the live call),
- **Respond**: a calm counter-reply (used in text mode).

The English/Hindi training pairs are in the bundle. Step 1 machine-translates a sample into
Hindi, Tamil, Telugu, Malayalam and Roman Urdu with IndicTrans2 ("silver" data) so Model B
learns every language. If that step fails, training still runs on English + Hindi.

## WHAT YOU DO
1. Unzip `civitas_model_b.zip` on your PC and upload the **`civitas_model_b`** folder to
   **My Drive** (or upload the zip itself to My Drive).
2. Optional, for the silver step: on huggingface.co open
   `ai4bharat/indictrans2-en-indic-dist-200M`, accept its terms if asked, create a read token
   (Settings → Access Tokens) and add it in Colab under **🔑 Secrets** as `HF_TOKEN`.
3. Optional: if Model A is trained, leave `civitas_model_a/runs/model_a/best_model.pt` on Drive;
   this notebook then also measures how often Model B's outputs are safe.
4. Open `civitas_model_b/train_model_b.ipynb` with Colab → **Runtime → T4 GPU → Run all**.
   About 30–60 min of translation, then 1.5–3 h of training. Re-run after a disconnect:
   finished languages and training steps are skipped.
5. Download from `civitas_model_b/runs/model_b/`: **`civitas_results_model_b.zip`** and
   **`model_b_adapter.zip`**, and give both to Claude (zip → `artifacts/`, adapter unzipped
   → `artifacts/model_b/adapter/`).
6. Do **not** tick CONFIRM in the last cell until Claude says tuning is finished.
'''),
    code('''
import subprocess, torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU."
GPU = torch.cuda.get_device_name(0)
major, _ = torch.cuda.get_device_capability(0)
BATCH, ACCUM = (32, 1) if "A100" in GPU else (16, 2)
print(GPU, "| precision:", "bf16" if major >= 8 else "fp32 (mT5 overflows in fp16 on T4)",
      f"| batch {BATCH} x accum {ACCUM}")
'''),
    setup("civitas_model_b"),
    code('''
# Pinned versions: IndicTrans2's remote code needs transformers 4.4x; peft 0.13 matches it.
!pip -q install "transformers==4.46.3" "peft==0.13.2" sentencepiece sacrebleu pyarrow "indic-transliteration>=2.3.45"
r = subprocess.run("pip -q install IndicTransToolkit", shell=True)
SILVER_TOOLKIT = r.returncode == 0
print("IndicTransToolkit installed:", SILVER_TOOLKIT)
try:
    from google.colab import userdata
    from huggingface_hub import login
    login(userdata.get("HF_TOKEN"))
except Exception as e:
    print("No HF_TOKEN secret (fine if the model is not gated):", type(e).__name__)
'''),
    code('''
# Step 1: silver translations. One language at a time, saved to Drive, skipped when done.
DO_SILVER = True   #@param {type:"boolean"}
PER_TASK = 2500    #@param {type:"integer"}
LANGS = ["hi", "ta", "te", "ml", "ur_roman"]
import time
import pandas as pd
SILVER_DIR = f"{RUNS}/silver"
PARTS = f"{SILVER_DIR}/parts"
os.makedirs(PARTS, exist_ok=True)
SILVER = f"{SILVER_DIR}/silver_pairs.parquet"
STARTED = time.time()
if DO_SILVER and SILVER_TOOLKIT and not os.path.exists(SILVER):
    try:
        from model_b_generation.silver import IndicTrans2, INDICTRANS_TARGETS, make_silver_rows, select_english
        english = select_english(pd.read_parquet(f"{DATA}/pairs.parquet"), PER_TASK)
        print(english.groupby(["task", "split"]).size())
        mt, cache = None, {}
        for lang in LANGS:
            part = f"{PARTS}/{lang}.parquet"
            if os.path.exists(part):
                print(lang, "already done"); continue
            tgt = INDICTRANS_TARGETS[lang]
            if tgt not in cache:
                mt = mt or IndicTrans2()
                t0 = time.time()
                texts = pd.unique(pd.concat([english["source_text"], english["target_text"]])).tolist()
                cache[tgt] = dict(zip(texts, mt.translate(texts, tgt)))
                print(f"{lang}: {len(texts)} sentences in {time.time() - t0:.0f}s")
            tr = cache[tgt]
            make_silver_rows(english, lang, [tr[t] for t in english["source_text"]],
                             [tr[t] for t in english["target_text"]]).to_parquet(part, index=False)
            print(lang, "->", part)
        pd.concat([pd.read_parquet(f"{PARTS}/{l}.parquet") for l in LANGS],
                  ignore_index=True).to_parquet(SILVER, index=False)
        del mt; torch.cuda.empty_cache()
    except Exception as e:
        print("SILVER STEP FAILED - continuing with English + Hindi only:", repr(e))
print("silver data:", SILVER if os.path.exists(SILVER) else "none")
'''),
    code('''
# Step 2: final leak-free splits (gold pairs + silver translations, grouped by original sentence)
GEN = "/content/gen_data"
silver_arg = f"--silver {SILVER}" if os.path.exists(SILVER) else ""
!cd {WORK} && python -m model_b_generation.prepare_splits_gen --pairs {DATA}/pairs.parquet {silver_arg} --review none --out_dir {GEN}
shutil.copy(f"{DATA}/manifest.json", f"{GEN}/manifest.json")
'''),
    code('''
# Step 3: training config. Every key is a train_gen.py flag
RUN_NAME = "model_b"  #@param {type:"string"}
OUT = f"{RUNS}/{RUN_NAME}"
MODEL_A = next((p for r in ROOTS for p in glob.glob(f"{r}/civitas_model_a*/**/runs/model_a/best_model.pt",
                                                     recursive=True)), "")
CONFIG = {
    "data_dir": GEN,
    "output_dir": OUT,
    "base": "bigscience/mt0-base",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": BATCH, "grad_accum": ACCUM,
    "lr": 5e-4, "warmup_ratio": 0.05,
    "max_input_length": 128, "max_output_length": 64,
    "evals_per_epoch": 2, "eval_gen_rows": 600, "patience": 3,
    "save_every": 300,
    "sample_power": 0.5,
    "model_a_ckpt": MODEL_A,
}
print("safety metric:", "on (" + MODEL_A + ")" if MODEL_A else "off (no trained Model A on Drive)")
ARGS = " ".join(f"--{k} {v!r}" if isinstance(v, str) else f"--{k} {v}" for k, v in CONFIG.items())
print(ARGS)
'''),
    code('''
# Train. Safe to re-run after a disconnect (--resume)
!cd {WORK} && python -m model_b_generation.train_gen {ARGS} --resume
'''),
    md('''
## Results (saved to files)
Writes `runs/model_b/civitas_results_model_b.zip` (metrics, history, sample generations,
silver stats) and `runs/model_b/model_b_adapter.zip`.
'''),
    code('''
import results_io
if os.path.exists(SILVER):
    s = pd.read_parquet(SILVER)
    results_io.write_json(OUT, "silver_report.json", {
        "rows": len(s),
        "by_language_task": {f"{l}|{t}": int(n) for (l, t), n in s.groupby(["language", "task"]).size().items()},
        "examples": s.groupby("language").head(3)[["language", "task", "en_source_text", "source_text",
                                                   "en_target_text", "target_text"]].to_dict("records")})
m = json.load(open(f"{OUT}/results/metrics.json"))
print({k: m.get(k) for k in ("chrf", "copy_rate", "safety_rate", "val_loss", "selection_score", "global_step")})
for g, v in m.get("by_group", {}).items():
    print(f"  {g:<22} chrF {v['chrf']:>5}  copy {v['copy_rate']:.2f}  safety {v['safety_rate']}")
!cd {WORK} && python -m results_io --run_dir {OUT} --zip {OUT}/civitas_results_model_b.zip
shutil.make_archive(f"{OUT}/model_b_adapter", "zip", f"{OUT}/best")
print("download:", f"{OUT}/civitas_results_model_b.zip", "and", f"{OUT}/model_b_adapter.zip")
'''),
    md('''
---
# RUN ONCE — final report
Only after Claude confirms tuning is finished. Scores the **test** split once.
'''),
    code('''
CONFIRM = False  #@param {type:"boolean"}
assert CONFIRM, "Tick CONFIRM only when tuning is finished."
!cd {WORK} && python -m model_b_generation.evaluate_gen --data_dir {GEN} --run_dir {OUT} --model_a_ckpt "{MODEL_A}"
!cd {WORK} && python -m results_io --run_dir {OUT} --zip {OUT}/civitas_results_model_b.zip
print("download again:", f"{OUT}/civitas_results_model_b.zip")
'''),
]

NOTEBOOKS = {"train_model_a.ipynb": NB_A, "train_model_b.ipynb": NB_B}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        write(name, cells)
