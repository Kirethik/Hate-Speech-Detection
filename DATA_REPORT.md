# Data report

## Rows per split x language
| language | train | val | test |
|---|---|---|---|
| en | 55336 | 4959 | 4959 |
| hi | 26216 | 1616 | 1617 |
| kn | 30276 | 1871 | 1871 |
| ml | 30792 | 1700 | 1699 |
| ta | 56915 | 3523 | 3524 |
| te | 23427 | 1452 | 1453 |
| ur_roman | 6476 | 361 | 360 |

## Train: source x language
| source | en | hi | kn | ml | ta | te | ur_roman |
|---|---|---|---|---|---|---|---|
| dravidiancodemix | 0 | 0 | 4435 | 10771 | 33328 | 0 | 0 |
| dynahate | 11948 | 0 | 0 | 0 | 0 | 0 | 0 |
| hatexplain | 16251 | 0 | 0 | 0 | 0 | 0 | 0 |
| implicithate | 5670 | 0 | 0 | 0 | 0 | 0 | 0 |
| macd | 0 | 26216 | 25841 | 20021 | 23587 | 23427 | 0 |
| ruhsold | 0 | 0 | 0 | 0 | 0 | 0 | 6476 |
| sbic | 13438 | 0 | 0 | 0 | 0 | 0 | 0 |
| toxigen | 8029 | 0 | 0 | 0 | 0 | 0 | 0 |

## Train: hate_label and script per language
| language | hate=0 | hate=1 | latin | mixed | native |
|---|---|---|---|---|---|
| en | 22264 | 33072 | 55335 | 1 | 0 |
| hi | 12343 | 13873 | 0 | 1131 | 25085 |
| kn | 16680 | 13596 | 2919 | 2470 | 24887 |
| ml | 21202 | 9590 | 6543 | 2791 | 21458 |
| ta | 37627 | 19288 | 26793 | 1545 | 28577 |
| te | 11138 | 12289 | 0 | 1238 | 22189 |
| ur_roman | 3464 | 3012 | 6476 | 0 | 0 |

## Train: auxiliary label coverage per language
| language | target | severity | rationale |
|---|---|---|---|
| en | 35671 | 55336 | 9110 |
| hi | 0 | 0 | 0 |
| kn | 0 | 0 | 0 |
| ml | 0 | 0 | 0 |
| ta | 0 | 0 | 0 |
| te | 0 | 0 | 0 |
| ur_roman | 0 | 6476 | 0 |

## Train: target / severity class counts
| target | rows |
|---|---|
| none | 16302 |
| religion | 4527 |
| gender | 4522 |
| caste_ethnicity | 7271 |
| disability | 895 |
| political | 163 |
| nationality_migrant | 598 |
| other | 1393 |

| severity | rows |
|---|---|
| normal | 25728 |
| offensive_profanity | 8384 |
| hate | 27700 |

## Warnings
- target class `political` has only 163 train rows (< 200)
