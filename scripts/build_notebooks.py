"""
Generates the Colab notebooks from the cell text below, so notebook code is
reviewed and diffed like any other code. Edit here, then run:

    python scripts/build_notebooks.py

Two self-contained notebooks, one per bundle made by scripts/make_bundles.py:
    notebooks/train_model_a.ipynb  (inside civitas_model_a/)
    notebooks/train_model_b.ipynb  (inside civitas_model_b/)
Each finds its bundle folder on Google Drive, trains with resume-after-disconnect,
and ends by writing ONE file, runs/civitas_model_{a,b}_trained.zip (weights + results),
which `python scripts/install_models.py` drops straight into the project.
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
# Get the bundle (code + data). Works with ANY of these - no folder upload needed:
#   a) {bundle}.zip uploaded to My Drive (any name like "{bundle} (1).zip" is fine)
#   b) {bundle}.zip dragged into Colab's Files panel (left sidebar)
#   c) nothing uploaded yet: a "Choose files" button appears below - pick the zip
# Checkpoints and the final model file are saved to My Drive/civitas_runs/{bundle}/
import glob, hashlib, json, os, shutil, sys, zipfile
NAME = "{bundle}"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    MYDRIVE = "/content/drive/MyDrive"
except Exception as e:
    MYDRIVE = None
    print("WARNING: Google Drive not mounted - progress is LOST if Colab disconnects.", e)

def find_zip():
    places = ["/content"] + ([MYDRIVE, f"{{MYDRIVE}}/civitas", f"{{MYDRIVE}}/*"] if MYDRIVE else [])
    for d in places:
        hits = sorted(glob.glob(f"{{d}}/{{NAME}}*.zip"), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    for d in ([MYDRIVE, f"{{MYDRIVE}}/civitas", f"{{MYDRIVE}}/*"] if MYDRIVE else []):
        for c in glob.glob(f"{{d}}/{{NAME}}*/code") + glob.glob(f"{{d}}/{{NAME}}*/{{NAME}}/code"):
            return os.path.dirname(c)          # an unzipped folder on Drive also works
    from google.colab import files
    print(f"Pick {{NAME}}.zip from your computer:")
    up = files.upload()
    z = next((f for f in up if f.startswith(NAME) and f.endswith(".zip")), None)
    assert z, f"That was not {{NAME}}.zip - run this cell again and pick the right file."
    return f"/content/{{z}}"

src = find_zip()
BUNDLE = "/content/bundle"
if os.path.isdir(src):
    shutil.copytree(src, f"{{BUNDLE}}/{{NAME}}", dirs_exist_ok=True)
else:
    print("unzipping", src)
    zipfile.ZipFile(src).extractall(BUNDLE)
BUNDLE = f"{{BUNDLE}}/{{NAME}}"
WORK, DATA = "/content/work", "/content/data"
RUNS = f"{{MYDRIVE}}/civitas_runs/{{NAME}}" if MYDRIVE else f"/content/civitas_runs/{{NAME}}"
shutil.copytree(f"{{BUNDLE}}/code", WORK, dirs_exist_ok=True)
shutil.copytree(f"{{BUNDLE}}/data", DATA, dirs_exist_ok=True)
os.makedirs(RUNS, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

manifest = json.load(open(f"{{DATA}}/manifest.json"))
for name, digest in manifest["files"].items():
    h = hashlib.sha256(open(f"{{DATA}}/{{name}}", "rb").read()).hexdigest()
    assert h == digest, f"{{name}} is corrupted (sha256 mismatch) - upload the zip again"
print("bundle from:", src, "| saving runs to:", RUNS)
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
1. Do **not** unzip anything. Open https://colab.research.google.com → **File → Upload notebook**
   → pick `train_model_a.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** (L4/A100 if you have Pro), then **Runtime → Run all**.
   Allow Google Drive access when asked (that is where progress is saved).
3. When the cell "Get the bundle" shows a **Choose files** button, pick `civitas_model_a.zip`.
   (Or upload `civitas_model_a.zip` to My Drive beforehand; the notebook then finds it itself.)
4. If Colab disconnects: reconnect and **Run all** again (pick the zip again if asked). Training resumes from the last save
   (at most ~500 steps are repeated).
5. When it finishes, download the one file **My Drive → `civitas_runs/civitas_model_a/civitas_model_a_trained.zip`**
   (trained weights + all results) into your Civitas project folder and run
   `python scripts/install_models.py --check`. Do not unzip it by hand; the script puts
   every file where the app expects it.
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
the single file to download: My Drive → `civitas_runs/civitas_model_a/civitas_model_a_trained.zip`.
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
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {RUNS}/civitas_model_a_trained.zip
print("DOWNLOAD THIS ONE FILE:", f"{RUNS}/civitas_model_a_trained.zip")
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
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {RUNS}/civitas_model_a_trained.zip
print("DOWNLOAD AGAIN (now includes the test report):", f"{RUNS}/civitas_model_a_trained.zip")
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
1. Do **not** unzip anything. Open https://colab.research.google.com → **File → Upload notebook**
   → pick `train_model_b.ipynb`. When the "Get the bundle" cell shows **Choose files**, pick
   `civitas_model_b.zip` (or upload that zip to My Drive beforehand).
2. Optional, for the silver step: on huggingface.co open
   `ai4bharat/indictrans2-en-indic-dist-200M`, accept its terms if asked, create a read token
   (Settings → Access Tokens) and add it in Colab under **🔑 Secrets** as `HF_TOKEN`.
3. Optional: if Model A is trained, its checkpoint stays in My Drive → `civitas_runs/civitas_model_a/`;
   this notebook then also measures how often Model B's outputs are safe.
4. **Runtime → Change runtime type → T4 GPU → Run all**.
   About 30–60 min of translation, then 1.5–3 h of training. Re-run after a disconnect:
   finished languages and training steps are skipped.
5. When it finishes, download the one file **My Drive → `civitas_runs/civitas_model_b/civitas_model_b_trained.zip`**
   (LoRA adapter + all results) into your Civitas project folder and run
   `python scripts/install_models.py --check`.
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
MODEL_A = f"{os.path.dirname(RUNS)}/civitas_model_a/model_a/best_model.pt"
MODEL_A = MODEL_A if os.path.exists(MODEL_A) else ""
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
Writes the single file to download: My Drive → `civitas_runs/civitas_model_b/civitas_model_b_trained.zip` (adapter, metrics,
history, sample generations, silver stats).
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
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_b --zip {RUNS}/civitas_model_b_trained.zip
print("DOWNLOAD THIS ONE FILE:", f"{RUNS}/civitas_model_b_trained.zip")
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
!cd {WORK} && python -m results_io --run_dir {OUT} --kind model_b --zip {RUNS}/civitas_model_b_trained.zip
print("DOWNLOAD AGAIN (now includes the test report):", f"{RUNS}/civitas_model_b_trained.zip")
'''),
]

NOTEBOOKS = {"train_model_a.ipynb": NB_A, "train_model_b.ipynb": NB_B}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        write(name, cells)
