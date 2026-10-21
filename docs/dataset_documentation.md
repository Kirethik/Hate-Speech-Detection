# Civitas AI — Dataset Documentation

## Overview

Civitas AI leverages curated, multi-source benchmarks covering English, Hindi, Roman-Urdu, and Dravidian languages (Tamil, Telugu, Malayalam, Kannada).

## Data Sources

1. **HateXplain** (Mathew et al., 2021)
   - Scope: English social media posts annotated for hate, normal, offensive, target community, and word-level rationale spans.
2. **DynaHate** (Vidgen et al., 2021)
   - Scope: Dynamic adversarial hate speech rounds with contrastive perturbed examples.
3. **HASOC 2019-2021** (Modha et al.)
   - Scope: Indo-Aryan and Dravidian multilingual hate speech (Hindi, Marathi, Tamil, Malayalam).
4. **MACD** (Chakravarthi et al., 2021)
   - Scope: Multi-aspect code-mixed Dravidian social commentary.
5. **RUHSOLD** (Rizwan et al., 2020)
   - Scope: Roman Urdu hate speech and offensive language classification.
6. **ParaDetox & CONAN** (Logacheva et al., 2022; Chung et al., 2019)
   - Scope: Parallel constructive rewrites and expert counter-narrative responses for Model B training.

## Leakage Prevention Protocol

- Deduplication by exact and normalized string matches.
- Strict disjoint splits: no training sentence appears in validation or test.
- Checkpoint selection is strictly evaluated on `validation.csv`. The `test.csv` split is reserved for final unbiased reporting.
