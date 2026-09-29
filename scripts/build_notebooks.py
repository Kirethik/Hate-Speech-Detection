"""
Generates the Colab notebooks from the cell text below, so notebook code is
reviewed and diffed like any other code. Edit here, then run:

    python scripts/build_notebooks.py

Every notebook: runs top-to-bottom, keeps state on Google Drive, resumes after
a disconnect, and ends by writing results/ + a civitas_results_<run>.zip that
you download into artifacts/ (no copy-pasting of cell output).
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


# --------------------------------------------------------------------------- shared cells
DRIVE = code('''
# Google Drive: everything that must survive a disconnect lives here
from google.colab import drive
drive.mount("/content/drive")
PROJECT = "/content/drive/MyDrive/civitas"
import os
os.makedirs(PROJECT, exist_ok=True)
print("project folder:", PROJECT, "->", sorted(os.listdir(PROJECT)))
''')

GET_CODE = code('''
# Get the code: from GitHub (fill REPO_URL) OR from civitas_code.zip in the Drive folder
REPO_URL = ""  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
import os, subprocess, zipfile, sys
CODE = "/content/civitas"
if REPO_URL:
    if os.path.exists(f"{CODE}/.git"):
        subprocess.run(["git", "-C", CODE, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, CODE], check=True)
else:
    # made on your PC with `make package-code`
    zpath = f"{PROJECT}/civitas_code.zip"
    assert os.path.exists(zpath), f"Set REPO_URL above, or upload civitas_code.zip to {PROJECT}"
    zipfile.ZipFile(zpath).extractall(CODE)
    inner = [d for d in os.listdir(CODE) if os.path.exists(f"{CODE}/{d}/config.py")]
    if not os.path.exists(f"{CODE}/config.py") and inner:  # zip had a top-level folder
        CODE = f"{CODE}/{inner[0]}"
os.chdir(CODE)
sys.path.insert(0, CODE)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout
      or "code from zip")
''')


def unpack(zip_name, dest):
    return code(f'''
# Unpack the data package and verify it against its manifest
import hashlib, json, zipfile, os
DATA = "{dest}"
zipfile.ZipFile(f"{{PROJECT}}/{zip_name}").extractall(DATA)
manifest = json.load(open(f"{{DATA}}/manifest.json"))
for name, digest in manifest["files"].items():
    h = hashlib.sha256(open(f"{{DATA}}/{{name}}", "rb").read()).hexdigest()
    assert h == digest, f"{{name}} is corrupted (sha256 mismatch) - re-upload the zip"
for split, info in manifest["splits"].items():
    print(f"{{split:>5}}: {{info['rows']:>7}} rows  languages={{info['language_counts']}}")
print("built from commit", manifest["git_commit"], "at", manifest["created_at"])
''')


# --------------------------------------------------------------------------- 02 Model A
NB02 = [
    md('''
# Civitas — train Model A (detector)

## WHAT YOU DO
1. On your PC, build the data (repo root):
   `python build_dataset.py`, then `python prepare_splits.py`, then `python scripts/package_data.py`.
   That creates `artifacts/civitas_data_v2.zip`.
2. Upload `civitas_data_v2.zip` to **Google Drive → My Drive → civitas/** (create the folder).
3. Get the code into Colab one of two ways:
   - push the repo to GitHub and paste its URL into the "Get the code" cell, **or**
   - run `make package-code` on your PC and upload `artifacts/civitas_code.zip` to the same Drive folder.
4. **Runtime → Change runtime type → T4 GPU** (L4/A100 if you have Pro), then **Runtime → Run all**.
5. If Colab disconnects: reconnect and **Run all** again. Training resumes from the last save
   on Drive (at most ~500 steps are repeated).
6. When it finishes, download from Drive → `civitas/model_a_v2/`:
   - `civitas_results_model_a.zip` → put it in `artifacts/` on your PC
   - `best_model.pt` → put it at `artifacts/model_a/best_model.pt`

   Then tell Claude "Model A results are in artifacts". Also keep `data/DATA_REPORT.md` handy.
7. Do **not** run the last section ("RUN ONCE") until Claude says tuning is finished.

Rough time on a free T4: 40–60 min per epoch for ~300k rows; the default 4 epochs with early
stopping usually ends in 2.5–4 hours, across one or two sessions.
'''),
    code('''
# 1. GPU check: picks batch size / precision defaults for this GPU
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then run all again."
GPU = torch.cuda.get_device_name(0)
major, _ = torch.cuda.get_device_capability(0)
if "A100" in GPU:
    BATCH, ACCUM = 64, 1
else:  # T4 / L4 and anything else with ~15-24 GB
    BATCH, ACCUM = 32, 1
PRECISION = "bf16" if major >= 8 else "fp16 + GradScaler (T4 has no native bf16)"
print(f"{GPU}: batch {BATCH} x accum {ACCUM}, {PRECISION}")
'''),
    DRIVE,
    GET_CODE,
    code('''
# Install what training needs (Colab already has torch)
!pip -q install "transformers>=4.40" "indic-transliteration>=2.3.45" pyyaml pyarrow scikit-learn sentencepiece
import transformers, indic_transliteration
print("transformers", transformers.__version__)
'''),
    unpack("civitas_data_v2.zip", "/content/data_pkg"),
    code('''
# the packaged data_config.yaml carries the language sampling weights used for this data
if os.path.exists(f"{DATA}/data_config.yaml"):
    os.replace(f"{DATA}/data_config.yaml", f"{CODE}/data_config.yaml")
'''),
    code('''
# Training config. Every key is a train.py flag (python train.py --help lists them all)
RUN_NAME = "model_a_v2"  #@param {type:"string"}
OUT = f"{PROJECT}/{RUN_NAME}"
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
    "lang_weights": "data_config.yaml",
}
ARGS = " ".join(f"--{k} {v!r}" if isinstance(v, str) else f"--{k} {v}" for k, v in CONFIG.items())
print(ARGS)
'''),
    code('''
# Train. Safe to re-run after a disconnect: --resume continues from OUT/last.pt
!cd {CODE} && python train.py {ARGS} --resume
'''),
    md('''
## Results (saved to files)
The next cell scores the best checkpoint on **val**, then writes `OUT/results/` and
`OUT/civitas_results_model_a.zip`. Download that zip and `best_model.pt` (see step 6 at the top).
'''),
    code('''
# Summary of the best checkpoint (val only) + behavioural probe on val -> results zip
import json
m = json.load(open(f"{OUT}/best_metrics.json"))
keep = ["threshold", "hate_macro_f1", "hate_roc_auc", "severity_macro_f1", "target_macro_f1",
        "rationale_token_f1", "epoch", "global_step"]
print(json.dumps({k: m.get(k) for k in keep}, indent=1))
!cd {CODE} && python -c "import pandas as pd; pd.read_parquet('{DATA}/val.parquet').to_csv('/content/val.csv', index=False)"
!cd {CODE} && python -m eval.probe_model --checkpoint {OUT}/best_model.pt --eval_csv /content/val.csv --out {OUT}/robustness.json
!cd {CODE} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {OUT}/civitas_results_model_a.zip
print("download:", f"{OUT}/civitas_results_model_a.zip", "and", f"{OUT}/best_model.pt")
'''),
    md('''
---
# RUN ONCE — final report
**Only run this after Claude confirms tuning is finished.** The test split is the one honest
number you get. If you change anything (data, config, code) after looking at these results,
the test numbers stop being trustworthy.
'''),
    code('''
# Test split + HateCheck functional tests. RUN ONCE. Re-zips results with the test report.
CONFIRM = False  #@param {type:"boolean"}
assert CONFIRM, "Tick CONFIRM only when tuning is finished (see the note above)."
!cd {CODE} && python train.py --eval_only --split test --data_dir {DATA} --output_dir {OUT} --batch_size {BATCH}
!cd {CODE} && python -m eval.eval_hatecheck --checkpoint {OUT}/best_model.pt --out {OUT}/hatecheck.json
!cd {CODE} && python -m results_io --run_dir {OUT} --kind model_a --data_manifest {DATA}/manifest.json --zip {OUT}/civitas_results_model_a.zip
print("download again:", f"{OUT}/civitas_results_model_a.zip")
'''),
]

# --------------------------------------------------------------------------- 03a silver
NB03A = [
    md('''
# Civitas — silver translations for Model B (IndicTrans2)

Translates English Rewrite/Respond pairs into Hindi, Tamil, Telugu, Malayalam and
Roman Urdu (via Hindi, then romanised) so Model B learns every language.
Machine translation is imperfect: you review a sample afterwards.

## WHAT YOU DO
1. On your PC (repo root):
   `python -m model_b_generation.build_dataset_gen`, then
   `python -m model_b_generation.prepare_splits_gen`, then
   `python scripts/package_data.py --kind gen` → `artifacts/civitas_gen_data.zip`.
2. Upload `civitas_gen_data.zip` (and `civitas_code.zip` if you don't use GitHub) to **My Drive → civitas/**.
3. On huggingface.co, open `ai4bharat/indictrans2-en-indic-dist-200M` and accept its terms if
   it asks. Create a read token (Settings → Access Tokens) and add it in Colab under
   **🔑 Secrets** as `HF_TOKEN` (turn on notebook access).
4. **Runtime → T4 GPU → Run all.** About 30–60 min. Re-run after a disconnect: finished
   languages are skipped.
5. Download from Drive → `civitas/silver/`:
   - `silver_pairs.parquet` → `data/gen/silver_pairs.parquet`
   - `civitas_results_silver.zip` → `artifacts/`
6. Review ~100 pairs per language on your PC:
   `python -m model_b_generation.review_silver --language ta` (repeat for hi, te, ml, ur_roman).
7. Rebuild the splits with the silver data and re-package:
   `python -m model_b_generation.prepare_splits_gen --silver data/gen/silver_pairs.parquet`
   then `python scripts/package_data.py --kind gen`, upload the new zip, and run notebook 03.
'''),
    code('''
import subprocess, torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU."
print(torch.cuda.get_device_name(0))
'''),
    DRIVE,
    GET_CODE,
    code('''
# IndicTrans2's remote code targets transformers 4.x, so pin it here (this runtime only)
!pip -q install "transformers>=4.40,<4.50" sentencepiece pyarrow "indic-transliteration>=2.3.45"
!pip -q install IndicTransToolkit || pip -q install git+https://github.com/VarunGumma/IndicTransToolkit
from huggingface_hub import login
try:
    from google.colab import userdata
    login(userdata.get("HF_TOKEN"))
except Exception as e:
    print("No HF_TOKEN secret (fine if the model is not gated):", e)
'''),
    unpack("civitas_gen_data.zip", "/content/gen_data"),
    code('''
# Config
PER_TASK = 2500   #@param {type:"integer"}  English rows per task (rewrite / respond) to translate
LANGS = ["hi", "ta", "te", "ml", "ur_roman"]
OUT = f"{PROJECT}/silver"
PARTS = f"{OUT}/parts"
os.makedirs(PARTS, exist_ok=True)
import time, pandas as pd
from model_b_generation.silver import select_english, make_silver_rows, IndicTrans2, INDICTRANS_TARGETS
pairs = pd.concat([pd.read_parquet(f"{DATA}/{s}.parquet") for s in ("train", "val", "test")])
english = select_english(pairs, PER_TASK)
print(english.groupby(["task", "split"]).size())
STARTED = time.time()
'''),
    code('''
# Translate, one language at a time. Each finished language is saved to Drive and skipped on re-run.
mt = None
cache = {}   # Hindi translations are reused for Roman Urdu
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
    rows = make_silver_rows(english, lang, [tr[t] for t in english["source_text"]],
                            [tr[t] for t in english["target_text"]])
    rows.to_parquet(part, index=False)
    print(lang, len(rows), "rows ->", part)
'''),
    code('''
# Combine -> silver_pairs.parquet + results zip
import results_io
silver = pd.concat([pd.read_parquet(f"{PARTS}/{l}.parquet") for l in LANGS], ignore_index=True)
silver.to_parquet(f"{OUT}/silver_pairs.parquet", index=False)
report = {
    "rows": len(silver),
    "by_language_task": {f"{l}|{t}": int(n) for (l, t), n in silver.groupby(["language", "task"]).size().items()},
    "by_split": silver["split"].value_counts().to_dict(),
    "per_task": PER_TASK,
    "examples": silver.groupby("language").head(3)[["language", "task", "en_source_text", "source_text",
                                                    "en_target_text", "target_text"]].to_dict("records"),
}
results_io.write_json(OUT, "metrics.json", report)
results_io.write_config(OUT, {"per_task": PER_TASK, "langs": LANGS, "model": "ai4bharat/indictrans2-en-indic-dist-200M"},
                        f"{DATA}/manifest.json", CODE)
results_io.write_env(OUT, STARTED)
results_io.zip_results(OUT, f"{OUT}/civitas_results_silver.zip")
print({k: report[k] for k in ("rows", "by_language_task")})
print("download:", f"{OUT}/silver_pairs.parquet", "and", f"{OUT}/civitas_results_silver.zip")
'''),
]

# --------------------------------------------------------------------------- 03 Model B
NB03 = [
    md('''
# Civitas — train Model B (Rewrite + Respond generator)

`bigscience/mt0-base` + LoRA. Model B writes the non-hateful version of a flagged sentence
(Rewrite, used in the live call) and a calm counter-reply (Respond, used in text mode).

## WHAT YOU DO
1. Finish notebook 03a and the review, then upload the **re-packaged** `civitas_gen_data.zip`
   (the one built with `--silver`) to **My Drive → civitas/**.
2. Optional but recommended: if Model A is trained, its `best_model.pt` stays at
   `My Drive/civitas/model_a_v2/best_model.pt`; this notebook then also measures how often
   Model B's outputs are safe.
3. **Runtime → T4 GPU → Run all.** Roughly 1.5–3 h on a T4. Re-run after a disconnect: it
   resumes from `last/` on Drive.
4. Download from Drive → `civitas/model_b_v1/`:
   - `civitas_results_model_b.zip` → `artifacts/`
   - `model_b_adapter.zip` → unzip into `artifacts/model_b/adapter/`

   Then tell Claude "Model B results are in artifacts".
5. Do **not** run "RUN ONCE" until Claude says tuning is finished.
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
    DRIVE,
    GET_CODE,
    code('''
!pip -q install "transformers>=4.40" "peft>=0.10" sentencepiece sacrebleu pyarrow "indic-transliteration>=2.3.45"
import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
'''),
    unpack("civitas_gen_data.zip", "/content/gen_data"),
    code('''
# Training config. Every key is a train_gen.py flag
RUN_NAME = "model_b_v1"  #@param {type:"string"}
OUT = f"{PROJECT}/{RUN_NAME}"
MODEL_A = f"{PROJECT}/model_a_v2/best_model.pt"
CONFIG = {
    "data_dir": DATA,
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
    "model_a_ckpt": MODEL_A if os.path.exists(MODEL_A) else "",
}
print("safety metric:", "on" if CONFIG["model_a_ckpt"] else "off (no Model A checkpoint on Drive)")
ARGS = " ".join(f"--{k} {v!r}" if isinstance(v, str) else f"--{k} {v}" for k, v in CONFIG.items())
print(ARGS)
'''),
    code('''
# Train. Safe to re-run after a disconnect (--resume)
!cd {CODE} && python -m model_b_generation.train_gen {ARGS} --resume
'''),
    md('''
## Results (saved to files)
Writes `OUT/civitas_results_model_b.zip` (metrics, history, samples) and `OUT/model_b_adapter.zip`.
'''),
    code('''
import json, shutil
m = json.load(open(f"{OUT}/results/metrics.json"))
print({k: m.get(k) for k in ("chrf", "copy_rate", "safety_rate", "val_loss", "selection_score", "global_step")})
for g, v in m.get("by_group", {}).items():
    print(f"  {g:<22} chrF {v['chrf']:>5}  copy {v['copy_rate']:.2f}  safety {v['safety_rate']}")
!cd {CODE} && python -m results_io --run_dir {OUT} --zip {OUT}/civitas_results_model_b.zip
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
!cd {CODE} && python -m model_b_generation.evaluate_gen --data_dir {DATA} --run_dir {OUT} --model_a_ckpt "{CONFIG['model_a_ckpt']}"
!cd {CODE} && python -m results_io --run_dir {OUT} --zip {OUT}/civitas_results_model_b.zip
print("download again:", f"{OUT}/civitas_results_model_b.zip")
'''),
]


if __name__ == "__main__":
    write("02_train_model_a.ipynb", NB02)
    write("03a_translate_silver.ipynb", NB03A)
    write("03_train_model_b.ipynb", NB03)
