# Label Mappings

## Target Mappings

| Source | Raw Label | Unified Target Label |
|--------|-----------|-----------------------|
| HateXplain | Islam, Jewish, Christian, Hindu, Buddhism, Nonreligious | religion |
| HateXplain | African, Caucasian, Asian, Hispanic, Arab, Indian, Indigenous, Minority | caste_ethnicity |
| HateXplain | Women, Men, Homosexual, Heterosexual, Bisexual, Asexual | gender |
| HateXplain | Refugee | nationality_migrant |
| HateXplain | Disability | disability |
| HateXplain | Economic, Other | other |
| HateXplain | None | none |
| DravidianCodeMix | Individual | none |
| DravidianCodeMix | Group, Other | other |
| DravidianCodeMix | None | none |
| SBIC | women, men, lgbtq, gay | gender |
| SBIC | black, white, asian, hispanic, latinx | caste_ethnicity |
| SBIC | jewish, muslim, christian | religion |
| SBIC | refugee, immigrant | nationality_migrant |
| SBIC | disabled, disability | disability |
| ToxiGen | black, asian, native_american, latino | caste_ethnicity |
| ToxiGen | jewish, muslim | religion |
| ToxiGen | lgbtq, women | gender |
| ToxiGen | mental_dis, physical_dis | disability |

## Severity Mappings

| Source | Raw Label | Unified Severity Label |
|--------|-----------|------------------------|
| HASOC | HATE | hate |
| HASOC | OFFN, PRFN | offensive_profanity |
| HASOC | NONE, NOT | normal |
| RUHSOLD | Hate Speech, Extreme | hate |
| RUHSOLD | Abusive, Offensive | offensive_profanity |
| RUHSOLD | Normal | normal |

## Ambiguous Mappings for User Decision

1. **HASOC Task 1 HOF:** Currently maps to `None` because it can be hate or offensive. Should we map it to `offensive_profanity` as a safe fallback if Task 2 labels are missing?
2. **DravidianCodeMix Group:** Currently maps to `other` since there's no finer granularity. Is there a better heuristic?
3. **Implicit Hate Corpus:** All hateful labels map to `hate`. Are there implicit forms that should map to `offensive_profanity`?
