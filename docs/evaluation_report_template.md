# Civitas AI — Model Evaluation Report Template

Use this template to document the quantitative and qualitative findings for Model A and Model B.

## 1. Model A (Hate Speech Detector)

### Metadata
- Date: YYYY-MM-DD
- Checkpoint: `model_artifacts/hate_detector/best_model.pt`
- Base Model: `FacebookAI/xlm-roberta-base`
- Evaluated Dataset: `datasets/processed/detector/test.csv` (N = X samples)

### Quantitative Performance
| Metric | Value | Baseline Comparison |
|---|---|---|
| Accuracy | 0.XX | 0.XX |
| Macro-F1 | 0.XX | 0.XX |
| Weighted-F1 | 0.XX | 0.XX |
| False Positive Rate (FPR) | 0.XX | 0.XX |
| False Negative Rate (FNR) | 0.XX | 0.XX |

### Confusion Matrix
| Actual \ Predicted | Predicted Non-Hateful | Predicted Hateful |
|---|---|---|
| **Actual Non-Hateful** | TN | FP |
| **Actual Hateful** | FN | TP |

---

## 2. Model B (Constructive Alternative Generator)

### Metadata
- Date: YYYY-MM-DD
- Checkpoint: `model_artifacts/alternative_generator/`
- Base Model: `google/mt5-small` / `bigscience/mt0-base`
- Evaluated Dataset: `datasets/processed/generator/test.csv` (N = X pairs)

### Quantitative Metrics
| Metric | Value |
|---|---|
| SacreBLEU | XX.X |
| ROUGE-L | XX.X |
| Model A Safety Verification Rate | XX.X% |

### Qualitative Samples
| Original Text | Reference Alternative | Generated Rewrite (Calm) | Safety Cleared? |
|---|---|---|---|
| Sample 1 | ... | ... | Yes / No |
| Sample 2 | ... | ... | Yes / No |
