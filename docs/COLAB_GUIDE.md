# Colab guide

All real training runs in Google Colab. Each model ships as one self-contained bundle:
the training code, every dataset that model needs, and its notebook.

| Bundle | Trains | GPU | Rough time |
|---|---|---|---|
| `artifacts/civitas_model_a.zip` | Model A, the detector (XLM-R, 4 heads) | T4 is enough | 3–4 h on T4 · ~1.5 h on L4 |
| `artifacts/civitas_model_b.zip` | Model B, Rewrite + Respond (mt0-base + LoRA), incl. IndicTrans2 silver translations | T4 is enough | 0.5–1 h translation + 1.5–3 h training |

Train Model A first if you can: Model B's notebook then also measures how often its outputs are safe.

## Making the bundles (on the PC, repo root)

```
make data          # Model A: build_dataset.py + prepare_splits.py -> data/{train,val,test}.csv
make gen-data      # Model B: converters + leak-free splits -> data/gen/
make bundles       # -> artifacts/civitas_model_a.zip, artifacts/civitas_model_b.zip
```

Raw data lives in `data/raw/` (see `docs/SOURCES.md`). The notebooks are generated from
`scripts/build_notebooks.py`: edit that file, not the .ipynb.

## Training one bundle

1. Unzip the bundle and upload the folder (`civitas_model_a` or `civitas_model_b`) to **My Drive**.
   Uploading the zip itself to My Drive also works; the notebook unpacks it.
2. Open the notebook inside the folder (`train_model_a.ipynb` / `train_model_b.ipynb`) with Colab.
3. Runtime → Change runtime type → **T4 GPU** → **Run all**. Allow Drive access.
4. Checkpoints and results go to `<bundle folder>/runs/` on Drive.

## When Colab disconnects

Reconnect, then **Run all** again. Training always passes `--resume` and continues from the last
save on Drive; Model B's translation step skips languages that are already done.
To start a fresh run instead, change `RUN_NAME` in the config cell.

## What comes back: files, not pasted output

Each run writes `runs/<run>/results/` and zips it (checkpoints are never inside the zip):

| File in `results/` | What it holds |
|---|---|
| `metrics.json` | the best checkpoint's val metrics (overall + per language / task) |
| `metrics_history.jsonl` | one line per evaluation during training |
| `config.json` | the exact flags, code commit and data-manifest hash |
| `env.json` | GPU, library versions, wall time |
| `samples.jsonl`, `silver_report.json` | Model B only: example generations, translation stats |
| `robustness.json` | Model A only: identity-swap and obfuscation probe on val |
| `test_report.json`, `hatecheck_report.json` | only after the RUN ONCE cell |

Download into `artifacts/` on the PC:
- Model A: `civitas_results_model_a.zip`, and `best_model.pt` → `artifacts/model_a/best_model.pt`
- Model B: `civitas_results_model_b.zip`, and `model_b_adapter.zip` unzipped → `artifacts/model_b/adapter/`

Then `make ingest` (or just tell Claude the files are there).

## Rules that keep the numbers honest

- **Do not tick CONFIRM in the "RUN ONCE" cell** until Claude says tuning is finished. It scores the
  test split (and HateCheck for Model A). Any change after looking at those numbers makes them optimistic.
- Checkpoints and thresholds are chosen on `val` only. Training never reads `test`.
- Augmentation and silver translations never cross splits: a translation always sits in the
  same split as the sentence it was translated from.
