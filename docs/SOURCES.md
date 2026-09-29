# Civitas AI — Dataset Registry

All raw data goes in `data/raw/<name>/` and is gitignored.
Never commit raw dataset files. Only converters and this table belong in the repo.

| Name | Languages | Local path | Official URL | License | Labels available | Status |
|---|---|---|---|---|---|---|
| HateXplain | en | `data/raw/HateXplain/` | https://github.com/hate-alert/HateXplain | CC BY 4.0 | hate, target, severity, rationale | ✅ Converter done |
| MACD | hi, ta, te, kn, ml | `data/raw/MACD/` | https://github.com/ShareChatAI/MACD | CC BY 4.0 | hate | ✅ Converter done |
| RUHSOLD | ur_roman | `data/raw/RUHSOLD/` | https://huggingface.co/datasets/community-datasets/roman_urdu_hate_speech | CC BY 4.0 | hate, severity | ✅ Converter done |
| DravidianCodeMix | ta, kn, ml | `data/raw/DravidianCodeMix-Dataset/` | https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset | TODO(user) | hate | ✅ Converter done |
| Latent Hatred (ImplicitHate) | en | HuggingFace cache | https://huggingface.co/datasets/SALT-NLP/ImplicitHate | TODO(user) | hate | ✅ Converter done |
| ToxiGen | en | HuggingFace cache | https://huggingface.co/datasets/skg/toxigen-data | MIT | hate, target | ✅ Converter done |
| SBIC | en | HuggingFace cache | https://huggingface.co/datasets/allenai/social_bias_frames | CC BY 4.0 | hate, target | ✅ Converter done |
| DynaHate | en | HuggingFace cache | https://huggingface.co/datasets/aps/dynahate | CC BY 4.0 | hate | ✅ Converter done |
| HASOC 2019/2020/2021 | en, hi | `data/raw/hasoc/` | https://hasocfire.github.io/hasoc/ | TODO(user) — check FIRE terms | hate, severity, rationale (2021) | ⏳ Converter written, needs download |
| DravidianLangTech (hate/offensive tracks) | ta, ml, te | `data/raw/dravidianlt/` | https://github.com/bharathichezhiyan/DravidianLangTech | TODO(user) | hate | ⏳ Converter written, needs download |
| CONSTRAINT 2021 Hindi Hostility | hi | `data/raw/constraint2021/` | https://github.com/constraint-2021/hostility-detection | TODO(user) | hate, severity | ⏳ Converter written, needs download |
| IEEE DataPort (Razi & Ejaz) | ur, ur_roman | `data/raw/ieee_razi/` | https://ieee-dataport.org/open-access/roman-urdu-hate-speech-dataset | TODO(user) — needs IEEE account | hate | ⏳ Converter written, needs download |
| HateCheck | en | `data/raw/hatecheck/` | https://github.com/paul-rottger/hatecheck-data | CC BY 4.0 | functional test labels | ✅ TEST-ONLY, never trains |
| Multilingual HateCheck | en, hi | `data/raw/multilingual_hatecheck/` | https://github.com/paul-rottger/multilingual-hatecheck | CC BY 4.0 | functional test labels | ✅ TEST-ONLY, never trains |

---

## Model B — Alternate Speech Datasets

Built by `python -m model_b_generation.build_dataset_gen` (prints rows per source), then
`python -m model_b_generation.prepare_splits_gen`. A missing source is skipped with a warning.

| Name | Languages | Local path (or HF id) | Official URL | License | Use | Status |
|---|---|---|---|---|---|---|
| ParaDetox | en | HF `s-nlp/paradetox`, or `data/raw/paradetox/*.csv` | https://huggingface.co/datasets/s-nlp/paradetox | TODO(user) — check the dataset card | Rewrite | ⏳ Converter written, verify columns |
| TextDetox 2024 | en, hi | HF `textdetox/multilingual_paradetox` | https://huggingface.co/datasets/textdetox/multilingual_paradetox | TODO(user) | Rewrite | ⏳ Converter written, verify columns |
| Multitarget-CONAN | en | HF `Rhma/Multitarget-CONAN`, or `data/raw/multitarget_conan/Multitarget-CONAN.csv` | https://github.com/marcoguerini/CONAN | TODO(user) | Respond | ✅ Columns checked (INDEX, HATE_SPEECH, COUNTER_NARRATIVE, TARGET, VERSION) |
| CONAN | en (EN rows of a multilingual set) | `data/raw/conan/CONAN.csv` or `CONAN.json` | https://github.com/marcoguerini/CONAN | TODO(user) | Respond | ⏳ Converter written, verify columns |
| Qian et al. 2019 (Reddit/Gab interventions) | en | `data/raw/qian_counter/{reddit,gab}.csv` | TODO(user) — the paper's GitHub repo | TODO(user) | Respond | ⏳ Converter written, verify columns |
| IndicCONAN | hi, en | `data/raw/indic_conan/*.csv` | TODO(user) | TODO(user) | Respond | ⏳ Converter written, verify columns |
| Silver translations (IndicTrans2) | hi, ta, te, ml, ur_roman | `data/gen/silver_pairs.parquet` (from notebook 03a) | https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M | TODO(user) — check the model licence | Rewrite + Respond | ⏳ Made by notebook 03a, reviewed with `review_silver` |

**Verify after downloading** (formats assumed, not checked):

| Source | Converter expects |
|---|---|
| ParaDetox / TextDetox | toxic + neutral column pairs named `en_toxic_comment`/`en_neutral_comment` or `toxic_sentence`/`neutral_sentence`; TextDetox has one split per language code (`en`, `hi`) |
| CONAN | a hate column (`hateSpeech`/`HATE_SPEECH`), a counter column (`counterSpeech`/`COUNTER_NARRATIVE`), and either a `language` column or `cn_id` starting with `EN` |
| Qian | `text` = numbered posts ("1. ...", one per line), `hate_speech_idx` = list of hateful post numbers, `response` = list of intervention strings |
| IndicCONAN | a column containing "hate", one containing "counter", optional `language` and `target` |

The earlier TER mini-corpus (synthetic template sentences) and the LT-EDI stub were removed:
templated pairs teach the model to parrot templates, and LT-EDI had no data.

---|---|---|---|---|---|---|
| CONAN | en (+EU) | HuggingFace cache | https://github.com/marcoguerini/CONAN | CC BY-NC 4.0 | Respond | ✅ Converter done |
| Multi-CONAN / Multitarget-CONAN | en | HuggingFace cache | TODO(user) | TODO(user) | Respond | ✅ Converter done |
| Gab & Reddit counter-speech (Qian et al.) | en | `data/raw/qian_counter/` | https://github.com/ziqizhang/iac_counter_speech | TODO(user) | Respond | ⏳ Converter written |
| ParaDetox | en | HuggingFace cache | https://huggingface.co/datasets/s-nlp/paradetox | Apache 2.0 | Rewrite | ⏳ Converter written |
| TextDetox 2024 | en, hi | HuggingFace cache | https://huggingface.co/datasets/textdetox/multilingual_paradetox | TODO(user) | Rewrite | ⏳ Converter written |

---

## Speech Evaluation Datasets (not for training)

| Name | Languages | What it is | Status |
|---|---|---|---|
| ADIMA (ShareChat, ICASSP 2022) | 10 Indic languages | Abusive audio clips — most on-target speech eval | TODO(user) — check access terms at ShareChat |
| MuTox (Meta) | multilingual | Multilingual audio toxicity | TODO(user) |
| DeToxy | en | English speech toxicity benchmark | TODO(user) |
| Kathbath / IndicVoices | hi, ta, te, ml | ASR WER benchmarks (not hate data) | TODO(user) |

---

## Downloading Instructions

For datasets marked `TODO(user)`:
1. Follow the official URL.
2. Read the license — confirm it covers your use (academic/demo/commercial differ).
3. Download into `data/raw/<name>/` exactly as shown in the Local path column.
4. Run `python build_dataset.py` — it will print which converters found data and which didn't.

---

## Verify after downloading (formats the converters assume but could not check)

`python build_dataset.py` prints rows, splits and languages for every source, and it stops if a
converter raises (use `--skip_failed` to build without that source). Check each of these:

| Source | Converter expects | Check |
|---|---|---|
| CONSTRAINT 2021 | `data/raw/constraint2021/{train,val}.csv`; text column `Post` or `text`; label column `Labels Set` or `label` with comma-separated values from non-hostile / fake / hate / offensive / defamation | The "dropped N rows" line lists only `fake` / `fake,non-hostile`, never labels spelt differently |
| DravidianLangTech | `data/raw/dravidianlt/{tamil,malayalam,telugu}/*{train,dev,test}*.tsv`, text TAB label | Row counts look right; hate=1 share is plausible (not ~100%) |
| HASOC | `data/raw/hasoc/<year>/...*{english,hindi,en,hi}*{train,test}*.tsv` with `text`, `task_1`, `task_2` | HASOC 2020 shipped .xlsx files; convert them to .tsv first |
| IEEE Razi | any csv/xlsx/tsv in `data/raw/ieee_razi/` with a tweet/text column and a label column | The printed `label_values` match the expected hate / not-hate values; unknown labels are dropped and listed |
| RUHSOLD | `data/raw/RUHSOLD/train.csv` with `tweet` and integer `label` 0..4 | All 5 labels are present |

Rows with an unrecognised label are dropped and reported, never guessed.
