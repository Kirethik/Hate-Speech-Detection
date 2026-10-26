# Civitas AI: Training Dataset

> **Content warning:** this dataset contains hateful, offensive, and abusive text collected from public research benchmarks. It is released for research on hate-speech detection and counter-speech only.

This branch holds only the data. The Civitas AI code (models, server, web UI) lives on the [`main`](https://github.com/Kirethik/Hate-Speech-Detection/tree/main) branch.

These are the exact data splits used to train the Civitas AI models:

- **Model A (hate-speech detector):** multilingual classification across 7 languages
- **Model B (constructive rewrite / counter-narrative generator):** paired input → target text

## Contents

```text
.
├── README.md
├── detector/
│   ├── train.csv          # 229,438 rows
│   ├── val.csv            #  15,482 rows
│   └── test.csv           #  15,483 rows
├── generator/
│   ├── train.csv          # 66,760 rows
│   ├── validation.csv     #  3,702 rows
│   └── test.csv           #  3,547 rows
├── DATA_REPORT.md         # per-language / per-source / label statistics
├── manifest.json          # row counts and SHA-256 hashes of the original splits
├── label_mapping.json     # class index mapping
└── data_config.yaml       # source caps and language weights used to build the data
```

## Detector schema (`detector/*.csv`)

| Column | Description |
|---|---|
| `text` | Input text |
| `language` | `en`, `hi`, `ta`, `te`, `ml`, `kn`, `ur_roman` |
| `hate_label` | `0` = not hateful, `1` = hateful |
| `target_label` | Target group (English sources only, where available) |
| `severity_label` | `normal`, `offensive_profanity`, `hate` (where available) |
| `rationale_spans` | Rationale spans (HateXplain rows) |
| `source` | Origin benchmark |
| `script` | `native`, `latin`, or `mixed` |

Rows per split and language:

| language | train | val | test |
|---|---|---|---|
| en | 55,336 | 4,959 | 4,959 |
| hi | 26,216 | 1,616 | 1,617 |
| kn | 30,276 | 1,871 | 1,871 |
| ml | 30,792 | 1,700 | 1,699 |
| ta | 56,915 | 3,523 | 3,524 |
| te | 23,427 | 1,452 | 1,453 |
| ur_roman | 6,476 | 361 | 360 |

## Generator schema (`generator/*.csv`)

| Column | Description |
|---|---|
| `input_text` | Original abusive or biased sentence |
| `target_text` | Constructive rewrite or counter-narrative |
| `language` | ISO language code |
| `source_dataset` | Origin benchmark |
| `task` | `rewrite` or `respond` |
| `tone` | Target tone |

Generator rows after filtering: **66,760** train, **3,702** validation, **3,547** test (74,009 total).

| Source | train | validation | test |
|---|---|---|---|
| paradetox | 15,911 | 926 | 882 |
| textdetox2024 | 720 | 33 | 45 |
| qian_reddit | 12,793 | 745 | 687 |
| qian_gab | 37,336 | 1,998 | 1,933 |

Model B was also trained on CONAN and Multitarget-CONAN pairs (7,865 train, 448 validation, 382 test rows). Those rows have been removed from this release because their license does not allow redistribution. The SHA-256 hashes and row counts in `manifest.json` refer to the original, unfiltered generator splits, so they will not match the files here.

## Sources and licenses

This is a **derived compilation**. All credit for the original text and annotations goes to the authors of the source datasets. Each source keeps its original license, so if you use the data, please cite the original papers and follow their terms.

| Source | Used in | License |
|---|---|---|
| HateXplain | detector | MIT |
| MACD | detector | see original release |
| DravidianCodeMix | detector | see original release |
| SBIC (Social Bias Frames) | detector | see original release |
| DynaHate | detector | see original release |
| ToxiGen | detector | see original release |
| ImplicitHate | detector | see original release |
| RUHSOLD | detector | see original release |
| ParaDetox | generator | see original release |
| TextDetox 2024 | generator | see original release |
| Qian et al. (Gab, Reddit) | generator | CC BY-NC 4.0 (non-commercial use only) |

The scripts that build these splits from the raw sources are on the `main` branch (`build_dataset.py`, `prepare_splits.py`, `datasets/prepare_datasets.py`).
