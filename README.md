# Civitas AI — Model A (Multilingual Detector)

XLM-R-base + shared encoder, four heads: hate (binary), target group, severity,
token rationale. Matches the architecture in your Slide 13.

## Verified

Runs end-to-end on the real `xlm-roberta-base` on a 6GB RTX 3050 (forward,
masked multi-task loss, backward, bf16 autocast, mid-epoch eval, early stop,
checkpoint save, one-shot test scoring). The split-hygiene guard in
`prepare_splits.py` asserts its own output is leak-free before writing, and
`train.py` re-checks at startup.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Convert your five sources to the unified schema

Required columns: `text, language, hate_label` (always).
Optional: `target_label, severity_label, rationale_spans` — leave a cell as
`-1` (labels) or `"[]"` (rationale_spans) if that source doesn't have it.
See `dataset.py` docstring for the full schema.

Per-source notes (map to your reference list):

| Source | Gives you | Watch out for |
|---|---|---|
| IEEE DataPort (Razi & Ejaz) | hate_label, language | needs account/subscription to download |
| RUHSOLD | hate_label, severity_label (5 fine classes → collapse to your 3) | Roman-Urdu only |
| MACD | hate_label | binary only, no target/severity — that's fine, it'll be masked |
| DravidianCodeMix | hate_label, target_label | target labels are Individual/Group/Other — remap to your 8-class scheme |
| HASOC 2019/20/21 | hate_label, severity_label, rationale_spans (2021 only) | rationale only exists for 2021 |
| HateXplain | hate_label, target_label, rationale_spans | English only — this is your rationale anchor |

Write one small `converters/<source>.py` per row in that table — each just
needs to read the raw format and emit rows matching the unified schema, then
concatenate everything into `data/train.csv` / `data/val.csv` (stratify the
split by `language` so no language is missing from validation).

## 2. Build trustworthy splits (required before training)

`build_dataset.py` only concatenates each converter's own train/val split. That
output is **not** safe to evaluate on, so run:

```bash
python prepare_splits.py
```

which rewrites `data/{train,val,test}.csv` after fixing three things measured in
the raw output:

| Problem found | Size | Why it matters |
|---|---|---|
| Text present in both train and eval | 1,321 rows (5.04% of the held-out pool, mostly DravidianCodeMix's overlapping train/dev files) | memorised rows inflate the score without improving the model |
| Duplicate rows inside train | 7,097 | over-weights whatever got duplicated |
| Texts labelled both 0 and 1 by different sources | 123 | contradictory supervision |
| No test set at all | — | `train.py` picks its checkpoint by val macro-F1, so reporting val means reporting the number you optimised against |

It splits the cleaned held-out pool 50/50 into `val` (checkpoint selection) and
`test` (reported once, at the end), stratified by source × language × label, and
asserts zero overlap between all three before writing. Originals are preserved
in `data/_original/`, so it is safe to re-run.

Dedup uses a loose key (casefold + strip every non-word character), because
social-media corpora are full of punctuation/emoji variants of the same string —
exact matching finds only 3.71% leakage where the loose key finds 5.04%.

## 3. Train

```bash
./run_training.sh          # prepares splits, then launches in the background
tail -f train.log          # progress
```

or directly:

```bash
python train.py --batch_size 16 --grad_accum 2 --epochs 4
```

`train.py` re-checks split disjointness at startup and **refuses to run on
leaked splits**. It selects on `val`, then scores the selected checkpoint on
`test` exactly once and writes `checkpoints/model_a/test_metrics.json`.

Notes on the defaults:

- **Effective batch 32 as 16×2 accumulation.** AdamW's fp32 moments for
  `xlm-roberta-base` already take ~4.4GB of a 6GB card, so a raw batch of 32
  risks an OOM hours into the run.
- **bf16 autocast** on Ampere and newer. bf16 keeps fp32's exponent range, so
  unlike fp16 it won't silently produce inf losses, and needs no GradScaler.
- **Dynamic padding.** Texts here average 29 tokens; the old fixed
  `padding="max_length"` at 128 spent ~78% of the GPU on padding.
- **`--hate_gamma 0`** (i.e. plain weighted cross-entropy) because hate is only
  57/43 imbalanced — focal down-weighting there mostly adds noise and hurts
  calibration. Severity keeps `gamma=2`, since it genuinely is imbalanced.
- The decision threshold is **fitted on val and applied unchanged to test**,
  rather than assuming 0.5.

### Reading the metrics honestly

`hate_label` is **not** hate speech. Every converter maps *offensive-or-worse*
to 1 (MACD abusive, DravidianCodeMix `Offensive_*`, HateXplain
offensive+hatespeech, RUHSOLD anything non-Normal). It is an **abuse/offensiveness**
detector; the *severity* head is what separates offensive from hate. Report it
that way.

Pooled macro-F1 is misleading here for two reasons, so the trainer always prints
the breakdowns instead:

- MACD is 58% of the eval set and carries the pooled average.
- Language correlates with the label (en 0.62 vs ml 0.28 hate rate), so a model
  can score respectably by learning "Malayalam → probably fine". Check
  `hate_by_language` and `hate_by_source`, not just the headline.
- `target_label` and `rationale_spans` come **only from HateXplain (English)**,
  so those two heads are English-only and will not transfer to the Indic
  languages. Target class `disability` has ~26 training examples — read
  `target_per_class`, never the target macro alone.

## 4. Results (run of 2026-08-05)

`python probe_model.py` re-checks the trained checkpoint behaviourally and
writes `checkpoints/model_a/robustness.json`.

| | val | test |
|---|---|---|
| hate macro-F1 | 0.8539 | **0.8519** |
| accuracy / ROC-AUC | — | 0.854 / 0.934 |

The **val→test gap of +0.002** is the number that makes the rest believable:
the checkpoint was chosen on val and test was scored once, so a large gap would
have meant the selection was overfitting. Decision threshold 0.71, fitted on val.

Per-source test macro-F1: macd 0.874, ruhsold 0.865, hatexplain 0.771,
dravidiancodemix 0.734 — all consistent with published baselines for these
corpora, which is what "realistic" looks like here. Per-language ranges
0.771 (en) to 0.901 (te); English is lowest because HateXplain's subtle hate
speech is genuinely the hardest slice, not because the model is weak there.

Aux heads (test): severity 0.650, rationale 0.748 token-F1, target 0.561 —
but see the dead-class note in `dataset.py`: target is **0.655** over the six
classes that actually have support.

### Behavioural findings the F1 does not show

- **No identity-mention shortcut.** 12/12 crafted sanity cases correct, 0/5
  identity-mention false positives. "i am a muslim woman and i am proud of my
  heritage" scores p(abuse)=0.010. This was the main thing worth disproving —
  abuse corpora train models to flag identity terms.
- **Activism sits near the boundary.** "black lives matter and we will keep
  marching" scores 0.645 against a 0.71 threshold — correct, but it would be a
  **false positive at the default 0.5**. The tuned threshold is doing real work.
- **Latin-script text is evadable.** Of the abuse the model catches, leetspeak
  (`a→@, e→3, o→0`) hides **59.1% of English** and **60.0% of Tamil**; spacing
  hides 12.6% overall. Flip rates are measured only over rows the transform
  actually altered — leetspeak is nearly a no-op on Devanagari/Kannada/Telugu,
  and counting those unchanged rows as "robust" understates the problem by 2×
  (it reports 14.5% instead of the true 31.6%).

## 4b. Implicit-hate work (the "Bronzites" failure)

### The failure

The 2026-08-05 model scored crude Tamil abuse at p=0.94 but rated

> "The Bronzites are a plague on every town they enter. Nothing good has ever
> come from trusting one of them."

at **p=0.28 — clean**. Two causes, both in the data rather than the architecture:

- **`hate_label` conflates offensive with hateful.** Every converter maps
  *offensive-or-worse* to 1, so the model optimised for "is this rude", and
  calm dehumanising rhetoric with no profanity is out of distribution.
- **The severity head, which is where that distinction belongs, was starved.**
  Only 22,727 of 190,354 train rows (11.9%) carried any severity label, and
  only **6,704 rows in the whole corpus** taught `severity == hate` — all of
  them from HateXplain (en) and RUHSOLD (ur_roman), none from any Dravidian
  source.

`eval_hatecheck.py` quantifies the gap independently: the baseline scores
**61.5%** on HateCheck against its 0.852 test macro-F1. Worst functionalities
are exactly the predicted ones — `derog_impl_h` (implicit derogation) 48.6%,
`counter_quote_nh` (counter-speech quoting hate) 30.1%, `negate_neg_nh` 31.6%,
`spell_space_add_h` 12.1%.

### Four fixes

| # | Fix | Where |
|---|---|---|
| 1 | Four English implicit-hate corpora added | `converters/{implicit_hate,toxigen,sbic,dynahate}.py` |
| 2 | Identity-term augmentation (swap + invented-group nonce) | `identity_augment.py`, `--identity_aug` |
| 3 | Dehumanisation-metaphor lexicon | `dehumanization.py` |
| 4 | NLI entailment second stage | `nli_stage.py`, `infer.py --nli` |

**Data.** Latent Hatred (ElSherief et al. 2021), ToxiGen (Hartvigsen et al.
2022), SBIC (Sap et al. 2020) and DynaHate (Vidgen et al. 2021) contribute hate
expressed *without* profanity. SBIC's `whoTarget` is the useful part: it
separates group-targeted from individual-targeted abuse, which is the
offensive-vs-hate axis itself. Net effect on the training split:

| | before | after |
|---|---|---|
| severity supervised | 22,727 (11.9%) | **61,812 (26.9%)** |
| `severity == hate` rows | 6,704 | **27,700** |
| sources teaching it | 2 | **6** |
| target supervised | 16,251 (8.5%) | 35,053 (15.3%) |

This also killed the dead-class problem the old README flagged: `political`
went 0 → 163 and `disability` 26 → 277, so no target class scores a mechanical
F1=0 any more and `TARGET_CLASSES` did not need changing.

The four sources are **English-only** — implicit-hate annotation essentially
does not exist for Tamil/Kannada/Telugu/Malayalam. They are capped (SBIC and
DynaHate at 15k each) so English grows from 16k to 55k of 229k rather than
taking over a multilingual model. Report this as a limitation: the improvement
is English-first by construction.

**Augmentation.** `identity_augment.py` rewrites the group noun and keeps the
label, in two modes. A within-category *swap* ("muslims"→"hindus") keeps
`target_label` valid; a *nonce* substitution ("muslims"→"Kelvarians") masks
`target_label` to -1, since an invented group has no real category. The nonce
mode is what teaches "<unknown group> are vermin is still hate" — the property
the Bronzites case needs. It runs as data *expansion* (229,438 → 242,604 rows),
not in-place rewriting, so `train.py` keeps tokenising once up front and holds
its throughput. `NONCE_GROUPS` deliberately excludes "Bronzites" so the probe
still measures generalisation rather than memorisation. Augmented rows are
re-checked against the val/test dedup keys and dropped on collision (7 were),
so augmentation cannot reintroduce the leakage `prepare_splits.py` removes.

**Lexicon.** Dehumanisation is lexically narrow (vermin / disease / animal /
filth / object frames, after Mendelsohn et al. 2020), so a lexicon catches it
regardless of which group is named. It requires the metaphor to be *predicated
of a plural human referent*, which is what separates "the Bronzites are a
plague" from "this game is cancer" — without that check it would be the same
keyword matching the model already does. It reports a human-auditable reason,
which a bare probability cannot:

```
$ python infer.py --text "The Bronzites are a plague on every town they enter."
  -> ABUSIVE  (p=0.287, threshold=0.71)  [escalated by lexicon]
     reason: dehumanising disease frame ('plague') applied to a group
```

**NLI.** `mDeBERTa-v3-base-mnli-xnli` scores entailment against group-agnostic
hypotheses ("This text describes a group of people as subhuman, vermin, or a
disease"), so it generalises compositionally to unseen groups. Consulted only
inside the uncertainty band (default `[0.25, 0.85)`) — where the detector is
confident it is usually right, and the band is where the old activism case sat
at 0.645 against a 0.71 threshold. Measured separation on the probe cases:
0.50–0.56 for hate, 0.003–0.15 for benign, so **`nli_threshold` needs fitting
near 0.40, not the 0.70 placeholder** — fit it on val the same way the
detector's own threshold is, and do not ship the default unfitted.

### Reproducing

```bash
python build_dataset.py          # now pulls the four HF sources too
python prepare_splits.py
./run_training.sh                # --identity_aug 1 is on by default
python eval_hatecheck.py --compare checkpoints/model_a/best_model_baseline.pt
python probe_model.py
pytest tests/ -q                 # 38 tests over the lexicon + augmenter
```

`build_dataset.py` now also refreshes `data/_original/`. Without that,
`prepare_splits.py` reads its stale first-run backup in preference to the new
build and the added sources never reach training.

**GPU note:** the card is 6 GB and training uses ~5.5 GB. An idle
`python infer.py` REPL holds ~2.3 GB and will kill a training run at its first
optimiser step. Pass `--device cpu` to the eval scripts while training runs.

## 5. Known gaps to fill before this is submission-ready

- `normalize_code_mixed()` in `dataset.py` is a **stub** — it only does
  whitespace collapsing. The 59%/60% leetspeak flip rates above are the direct,
  quantified cost of that: this is RQ1's de-obfuscation front-end, and the
  robustness probe is already wired to measure the improvement once you build
  it. Implement and unit-test it separately, then re-run `probe_model.py`.
- ~~**Retrain without the dead target classes.**~~ **Resolved by the implicit-hate
  data** (§4b): SBIC populates `political` (0 → 163) and ToxiGen/SBIC populate
  `disability` (26 → 277), so no class scores a mechanical F1=0 any more. Both
  are still thin — read `target_per_class`, not the target macro alone.
- Target/severity class label sets (`TARGET_CLASSES`, `SEVERITY_CLASSES` in
  `dataset.py`) are placeholders based on your slide language — confirm
  against your actual annotation guide once you've merged all five sources,
  since DravidianCodeMix's target scheme and HateXplain's target scheme
  don't line up 1:1 and you'll need an explicit remapping table.
- No perturbation/robustness eval loop yet (RQ1's flip-rate metric against
  the Style Perturbation dataset) — that's a separate `evaluate_robustness.py`
  you'll want once Model A trains cleanly.
