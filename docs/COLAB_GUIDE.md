# Colab guide

All real training runs in Google Colab. The notebooks are resumable: every checkpoint is written
to Google Drive, so a disconnect costs minutes, not the run.

| Notebook | What it does | GPU | Rough time |
|---|---|---|---|
| `02_train_model_a.ipynb` | Trains the detector (XLM-R, 4 heads) | T4 is enough | 2.5–4 h on T4 · ~1.5 h on L4 · ~45 min on A100 |
| `04_export.ipynb` | Optional ONNX int8 export of Model A | CPU runtime is fine | ~10 min |
| `03a_translate_silver.ipynb` / `03_train_model_b.ipynb` | Model B data + training | — | Added in Phase 3 |

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
8. If you use GitHub, paste the repo URL into cell 3. Then **Runtime → Run all** and allow Drive
   access when asked.
9. Leave the tab open. The log prints the loss every 100 steps and full validation metrics about
   3 times per epoch.

## When Colab disconnects

Reconnect, then **Runtime → Run all**. Cell 7 always passes `--resume`: training reloads
`My Drive/civitas/<RUN_NAME>/last.pt` (model, optimizer, schedule, position in the epoch) and
continues. At most ~500 steps are repeated. Nothing needs to be changed.

To start a fresh run instead, change `RUN_NAME` in cell 6.

## What to send back to Claude

1. The whole output of cell 8 ("Results to paste back"). It has the val metrics per language,
   script and source, the per-class target F1, and the identity and obfuscation probe.
2. `data/DATA_REPORT.md` from your PC.

Claude will diagnose the weak spots and propose at most three changes for the next run.

## Files to download to your PC afterwards

From `My Drive/civitas/<RUN_NAME>/` into `artifacts/model_a/`:
- `best_model.pt` (required; the local server loads it)
- `best_metrics.json`, `metrics_history.csv` (for the model card)
- after the export notebook: `export/model_a_int8.onnx` and the `export/tokenizer/` folder

`last.pt` is only for resuming. It is about 3× larger and not needed locally.

## Rules that keep the numbers honest

- **Do not run cell 9 ("RUN ONCE")** until Claude says tuning is finished. It scores the test
  split and HateCheck. Any change you make after looking at those numbers makes them optimistic.
- Checkpoints and the decision threshold are chosen on `val` only. Training never reads `test`.
- Augmentation (transliteration, typos, identity swaps) is applied to `train` only, and any
  augmented row that matches a val/test text is dropped.
