# Civitas AI — System Architecture

Civitas AI is a research-backed, production-grade system designed for real-time multilingual hate speech detection and context-aware constructive alternative generation.

```mermaid
flowchart TD
    subgraph Input_Processing [1. Input Processing]
        IN[User Input: Text or Audio] --> NORM[Script & Text Normalization]
    end

    subgraph Neural_Detector [2. Model A: Hate Speech Detector]
        NORM --> ENCODER[XLM-RoBERTa Shared Encoder]
        ENCODER --> H1[Binary Hate Classification Head]
        ENCODER --> H2[Target Group Identification Head: 8 classes]
        ENCODER --> H3[Severity Assessment Head: 3 classes]
        ENCODER --> H4[Token-Level Rationale Head: Character Spans]
    end

    subgraph Hybrid_Verification [3. Second-Stage Verification]
        H1 --> BAND{Uncertainty Band or Metaphor?}
        BAND -->|High Confidence| DECISION[Verdict & Attribution]
        BAND -->|Lexicon Match| LEX[Dehumanization Lexicon]
        BAND -->|Uncertainty Range| NLI[mDeBERTa Cross-Lingual NLI]
        LEX --> DECISION
        NLI --> DECISION
    end

    subgraph Alternative_Generation [4. Model B: Constructive Rewriter]
        DECISION -->|Hate / Offensive| COND[Prompt Conditioning: Tone + Target]
        COND --> SEQ2SEQ[mT5 / mt0 Seq2Seq Generator]
        SEQ2SEQ --> CAND[Candidate Generation: Rewrite & Respond]
    end

    subgraph Safety_Gating [5. Safety Clearance & Review]
        CAND --> GATE[Model A Safety Filter]
        GATE -->|Cleared| SAFE[Verified Constructive Alternatives]
        GATE -->|Flagged| FALLBACK[Masked Spans or Grounded Template]
        SAFE --> UI[Human Moderator / User Review Interface]
        FALLBACK --> UI
    end
```

## Core Architectural Principles

1. **Two Distinct Neural Networks**:
   - **Model A (Detector)**: `FacebookAI/xlm-roberta-base` fine-tuned for sequence and token classification across 8 languages.
   - **Model B (Generator)**: `google/mt5-small` / `bigscience/mt0-base` fine-tuned for conditional text rewriting with tone control.
2. **Safety Gating**:
   - Model B outputs are never automatically assumed to be safe. Every candidate generated is re-scored by Model A. Any candidate that repeats abusive vocabulary or scores above the safety threshold is dropped.
3. **Decoupled Lifecycle**:
   - Model A and Model B have completely separate training notebooks, configurations, tokenizers, checkpoints, and export artifacts.
   - Application development operates independently of training via a dedicated, transparent Demo Mode.
