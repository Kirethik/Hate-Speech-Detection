# Civitas AI — Model Evaluation Framework

Quantitative and qualitative evaluation tooling for both primary neural networks:
- **Model A (Detector)**: Evaluates multi-class hate classification, macro-F1, per-class metrics, false-positive/false-negative rates, and confusion matrices.
- **Model B (Generator)**: Evaluates meaning preservation, fluency, safety verification rate (Model A clearance), and tone conditioning across candidate rewrites.

## Usage

### Evaluate Model A (Detector)
```bash
python evaluation/detector_evaluation.py --data datasets/processed/detector/validation.csv --out evaluation/detector_eval_report.json --samples 200
```

### Evaluate Model B (Generator)
```bash
python evaluation/generator_evaluation.py --data datasets/processed/generator/validation.csv --out evaluation/generator_eval_report.json --samples 50
```
