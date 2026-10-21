# Civitas AI — Datasets

This directory contains dataset management scripts, schemas, manifests, and documentation for training Model A (Hate Speech Detector) and Model B (Constructive Alternative Generator).

## Directory Structure

```text
datasets/
├── README.md                 # This guide
├── manifest.json             # File hashes, split sizes, and source attributions
├── label_mapping.json        # Unified class index mapping for detector and generator
├── download_datasets.py      # Automated / documented download helpers
├── prepare_datasets.py       # Preprocessing, normalization, deduplication & splitting
├── raw/                      # Original, unaltered source data (gitignored)
│   └── README.md
├── processed/                # Standardized training splits
│   ├── detector/             # Model A training data
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── generator/            # Model B paired rewriting data
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
└── reports/                  # Generated statistical reports
    ├── dataset_summary.json
    └── label_distribution.json
```

## Supported Languages

- English (`en`)
- Hindi (`hi`)
- Urdu (`ur`) and Roman Urdu (`ur_roman`)
- Tamil (`ta`)
- Telugu (`te`)
- Malayalam (`ml`)
- Kannada (`kn`)

## Schemas

### 1. Detector (`processed/detector/*.csv`)
- `text`: Input text string
- `label`: Integer binary label (`0` = non_hateful, `1` = hateful)
- `label_name`: String representation (`non_hateful` or `hateful`)
- `language`: ISO language code
- `source_dataset`: Name of origin benchmark (e.g., `hatexplain`, `hasoc`, `macd`)
- `severity_label`: Optional ordinal severity (`0` = normal, `1` = offensive, `2` = hate)
- `target_label`: Optional target group category index
- `rationale_spans`: Optional JSON array of character offset spans

### 2. Generator (`processed/generator/*.csv`)
- `input_text`: Original abusive or biased sentence
- `target_text`: Constructive, human-reviewed rewrite or counter-narrative
- `language`: ISO language code
- `source_dataset`: Source dataset identifier (e.g., `paradetox`, `conan`, `indic_conan`)
- `task`: Task type (`rewrite` or `respond`)
- `tone`: Target tone (`calm`, `factual`, `firm`, `empathetic`, `professional`)
