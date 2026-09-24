# Label mappings

The code in `label_maps.py` is the source of truth; this page is the readable
version. `-1` means "unknown": the example still trains the hate head, and that
auxiliary head is masked out of the loss.

Classes:
- **Target (8):** none, religion, gender, caste_ethnicity, disability, political, nationality_migrant, other
- **Severity (3):** normal, offensive_profanity, hate

## Target

| Source | Raw value | Target |
|---|---|---|
| HateXplain | Islam, Jewish, Christian, Hindu, Buddhism, Nonreligious | religion |
| HateXplain | African, Caucasian, Asian, Hispanic, Arab, Indian, Indigenous, Minority | caste_ethnicity |
| HateXplain | Women, Men, Homosexual, Heterosexual, Bisexual, Asexual | gender |
| HateXplain | Refugee | nationality_migrant |
| HateXplain | Disability | disability |
| HateXplain | Economic, Other (and any unlisted value) | other |
| HateXplain | None | none |
| SBIC (`targetCategory`, offensive + group-targeted only) | race | caste_ethnicity |
| SBIC | gender | gender |
| SBIC | culture | religion |
| SBIC | disabled | disability |
| SBIC | social | **political** (the only source of this class) |
| SBIC | body, victim | other |
| SBIC | not offensive | none |
| ToxiGen (hateful rows) | black, asian, chinese, latino, mexican, native_american, middle_east | caste_ethnicity |
| ToxiGen | jewish, muslim | religion |
| ToxiGen | women, trans, lgbtq, bisexual | gender |
| ToxiGen | mental_dis\*, physical_dis\* | disability |
| ToxiGen | immigrant | nationality_migrant |
| ToxiGen | neutral rows | none |
| All other sources | — | -1 |

DravidianCodeMix / DravidianLangTech mark whether abuse targets an individual or a group, not which
identity group is targeted. That is a different axis, so their target is -1.

## Severity

| Source | Raw label | Severity | hate_label |
|---|---|---|---|
| HateXplain | hatespeech / offensive / normal (majority vote) | hate / offensive_profanity / normal | 1 / 1 / 0 |
| HASOC task 2 | HATE | hate | 1 |
| HASOC task 2 | OFFN, PRFN | offensive_profanity | 1 |
| HASOC | HOF with no task-2 label | offensive_profanity | 1 |
| HASOC | NOT / NONE | normal | 0 |
| RUHSOLD | Religious Hate, Sexism | hate | 1 |
| RUHSOLD | Abusive/Offensive, Profane/Untargeted | offensive_profanity | 1 |
| RUHSOLD | Normal | normal | 0 |
| CONSTRAINT 2021 (label set) | contains hate | hate | 1 |
| CONSTRAINT 2021 | contains offensive or defamation | offensive_profanity | 1 |
| CONSTRAINT 2021 | non-hostile | normal | 0 |
| CONSTRAINT 2021 | fake only, or an unknown label | **row dropped** | — |
| SBIC | offensive + group-targeted / offensive + individual / not offensive | hate / offensive_profanity / normal | 1 / 1 / 0 |
| ToxiGen, DynaHate | hate / not | hate / normal | 1 / 0 |
| Implicit Hate | implicit or explicit hate | hate | 1 |
| MACD, DravidianCodeMix, DravidianLangTech, IEEE Razi | binary only | -1 | from source |

## Decisions and caveats

- **CONSTRAINT "fake"** is misinformation, not abuse. Posts labelled only fake are dropped; posts that
  are also hate, offensive or defamatory keep the more severe label.
- **Defamation → offensive_profanity:** it attacks one person's reputation, not a protected group.
- **"not-&lt;language&gt;" rows** (DravidianCodeMix, DravidianLangTech) are a language judgement, not a
  hate judgement, so they are dropped.
- **Unknown labels** in CONSTRAINT and IEEE Razi are dropped and printed. They never default to hateful.
- **MACD** uses 0 = abusive, the inverse of our convention; the converter flips it.
- **`hate_label` really means "abusive or offensive"** for most sources. Show the severity head in the
  UI as the "hate vs. offensive" signal, not the binary head alone.
