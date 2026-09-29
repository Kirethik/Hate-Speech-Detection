# Colab guide

All real training runs in Google Colab. The notebooks are resumable: every checkpoint is written
to Google Drive, so a disconnect costs minutes, not the run.

| Notebook | What it does | GPU | Rough time |
|---|---|---|---|
| `02_train_model_a.ipynb` | Trains the detector (XLM-R, 4 heads) | T4 is enough | 2.5–4 h on T4 · ~1.5 h on L4 · ~45 min on A100 |
| `04_export.ipynb` | Optional ONNX int8 export of Model A | CPU runtime is fine | ~10 min |
| `03a_translate_silver.ipynb` | IndicTrans2 silver translations for Model B | T4 | 30–60 min |
| `03_train_model_b.ipynb` | Trains Model B (mt0-base + LoRA, Rewrite + Respond) | T4 is enough | 1.5–3 h on T4 |

The notebooks are generated from `scripts/build_notebooks.py`. Edit that file, not the .ipynb.

## Model A: step by step

**On your PC (repo root):**
1. Download the raw datasets into `data/raw/<name>/` as listed in `docs/SOURCES.md`.
2. Build the data: `python build_dataset.py`, then `python prepare_splits.py`.
   - The build stops on a converter error. Read the message; `--skip_failed` builds without that
     source.
   - Read `data/DATA_REPORT.md`. If Telugu or Malayalam still have fewer than ~3,000 train rows,
     find more data before spending Colab time.
3. Package the data: `make package-data` (or `python scripts/package_data.py`). This creates
   `artifacts/civitas_data_v2.zip`.
4. Package the code: push to GitHub, **or** run `make package-code`, which creates
   `artifacts/civitas_code.zip` from the committed code.

**In Google Drive:**

5. Create `My Drive/civitas/` and upload `civitas_data_v2.zip` (plus `civitas_code.zip` if you are
   not using GitHub).

**In Colab:**

6. Open `notebooks/02_train_model_a.ipynb` (File → Upload notebook, or open it from GitHub).
7. Runtime → Change runtime type → **T4 GPU** → Save.
8. If you use GitHub, paste the repo URL into the "Get the code" cell. Then **Runtime → Run all** and allow Drive
   access when asked.
9. Leave the tab open. The log prints the loss every 100 steps and full validation metrics about
   3 times per epoch.

## When Colab disconnects

Reconnect, then **Runtime → Run all**. The train cell always passes `--resume`: training reloads
`My Drive/civitas/<RUN_NAME>/last.pt` (model, optimizer, schedule, position in the epoch) and
continues. At most ~500 steps are repeated. Nothing needs to be changed.

To start a fresh run instead, change `RUN_NAME` in the config cell.

## What to send back to Claude: files, not pasted output

Every notebook ends by writing a `results/` folder next to its checkpoints on Drive and zipping
it (checkpoints are never inside the zip):

| File in `results/` | What it holds |
|---|---|
| `metrics.json` | the best checkpoint's val metrics (overall + per language / task) |
| `metrics_history.jsonl` | one line per evaluation during training |
| `config.json` | the exact flags, git commit and data-manifest hash |
| `env.json` | GPU, library versions, wall time |
| `samples.jsonl` | Model B only: 10 generations per task/language |
| `robustness.json` | Model A only: identity-swap and obfuscation probe on val |
| `test_report.json`, `hatecheck_report.json` | only after the RUN ONCE cell |

1. Download `civitas_results_<run>.zip` into `artifacts/` on your PC.
2. Run `python scripts/ingest_results.py` (it unpacks every zip into `artifacts/results/<run>/`
   and prints a summary), or just tell Claude the zip is there.
3. For Model A also keep `data/DATA_REPORT.md` handy.

Claude will diagnose the weak spots and propose at most three changes for the next run.

## Model B: step by step

1. On your PC: `python -m model_b_generation.build_dataset_gen`, then
   `python -m model_b_generation.prepare_splits_gen`, then `python scripts/package_data.py --kind gen`.
2. Upload `artifacts/civitas_gen_data.zip` to `My Drive/civitas/` and run **03a** (silver
   translations). Download `silver/silver_pairs.parquet` into `data/gen/`.
3. Review a sample per language: `python -m model_b_generation.review_silver --language ta`
   (repeat for hi, te, ml, ur_roman). Keys: k keep, d drop, f fix, s skip, q quit.
4. Rebuild with the reviewed silver data: `python -m model_b_generation.prepare_splits_gen
   --silver data/gen/silver_pairs.parquet`, re-package (`--kind gen`), upload again.
5. Run **03**. Download `civitas_results_model_b.zip` into `artifacts/` and unzip
   `model_b_adapter.zip` into `artifacts/model_b/adapter/`.

## Files to download to your PC afterwards

From `My Drive/civitas/<RUN_NAME>/` into `artifacts/model_a/`:
- `best_model.pt` (required; the local server loads it)
- `civitas_results_model_a.zip` into `artifacts/` (metrics for the model card)
- after the export notebook: `export/model_a_int8.onnx` and the `export/tokenizer/` folder

`last.pt` is only for resuming. It is about 3× larger and not needed locally.

## Rules that keep the numbers honest

- **Do not run the "RUN ONCE" cell** until Claude says tuning is finished. It scores the test
  split and HateCheck. Any change you make after looking at those numbers makes them optimistic.
- Checkpoints and the decision threshold are chosen on `val` only. Training never reads `test`.
- Augmentation (transliteration, typos, identity swaps) is applied to `train` only, and any
  augmented row that matches a val/test text is dropped.
