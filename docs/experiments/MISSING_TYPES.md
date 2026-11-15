# Controlled comparison: handling each type of missing value

## The five types of gaps

All counts are from the raw training file (256,351 rows; 6,291 DI and 250,060 IVF).
31 input columns have gaps: 29 numeric and 2 string. Several string columns also record
"unknown" as text instead of leaving a blank.

| Type | Meaning | Where | Evidence |
|---|---|---|---|
| 1. DI not applicable | DI creates no embryos, so embryo and egg fields cannot exist | 20 numeric columns plus `배아 생성 주요 이유`, empty for every DI row and never empty for IVF rows; `난자 출처 = 알 수 없음` on exactly the 6,291 DI rows | 100% vs 0% missing by treatment type |
| 2. Step not performed | An elapsed-day column times a step that did not happen | 5 elapsed-day columns (table below) | The gap lines up with the column recording the step |
| 3. Truly missing | The step happened but its day was not recorded | About 1,600 IVF rows across the same 5 columns; 2 rows of `특정 시술 유형` | Mismatches in the table below |
| 4. Blank in a flag that records only 1 | Probably "no" (assumed, not confirmed by the official data dictionary) | `착상 전 유전 검사 사용 여부`, `PGS 시술 여부`, `PGD 시술 여부` (99% blank) | No 0 is ever recorded |
| 5. Unknown | No structural explanation | `임신 시도 또는 마지막 임신 경과 연수` (96.3% blank, including 44,546 rows with a previous pregnancy); "알 수 없음" or "기록되지 않은 시행" in age, ovulation-induction and donor-age columns | — |

Elapsed-day columns among IVF rows:

| Column | Blank IVF rows | Of those, step not performed | Truly missing |
|---|---:|---|---:|
| `배아 이식 경과일` | 37,275 | 36,535 with `이식된 배아 수 = 0` | 740 |
| `배아 해동 경과일` | 209,691 | 209,689 with `동결 배아 사용 여부 = 0` | 2 |
| `난자 채취 경과일` | 51,197 | 51,158 with `수집된 신선 난자 수 = 0` | 39 |
| `난자 혼합 경과일` | 47,444 | 46,612 with `혼합된 난자 수 = 0` | 832 |
| `난자 해동 경과일` | 248,624 | 248,613 with `해동 난자 수 = 0` | 11 |

Rows with a blank transfer day have a 2.4% pregnancy rate, which fits "no embryo was
transferred".

## How each version handles them

| Type | Baseline (`baseline_cv.py`) | `native_nan` | `typed` rules (`missing_types_cv.py`) |
|---|---|---|---|
| 1. DI not applicable | Training-fold median; string gap becomes `__MISSING__` | Left as NaN | Counts and flags 0 on DI rows; elapsed days −1 |
| 2. Step not performed | Training-fold median | Left as NaN | −1, a value no real day can take |
| 3. Truly missing | Training-fold median, indistinguishable from type 2 | Left as NaN, indistinguishable from type 2 | Left as NaN, plus a `<column>_truly_missing` indicator |
| 4. Blank in a flag that records only 1 | **Training-fold median, which is 1, so every blank becomes 1** | Left as NaN | 0 |
| 5. Unknown | Median for numbers; text values stay as their own one-hot category | Same, but numbers left as NaN | Same as `native_nan` |

Every rule is either row-wise or fitted on the training part of each fold: the flag
columns (only value 1 recorded) and the DI columns (empty for every training DI row) are
chosen from training rows only. No label is used. `typed_median` applies the same rules
but fills the remaining gaps (types 3 and 5) with the training-fold median.

## Results

[Recorded run](../../results/missing_types_20261115T072026_459803Z/comparison.json). All
variants use the baseline's five folds; the first five use the baseline's untuned
LightGBM (250 trees).

| Variant | OOF ROC-AUC | Compared with | Pooled difference | Folds up |
|---|---:|---|---:|---:|
| `baseline` | 0.73922 | | | |
| `median_flags0`: only the type 4 bug fixed | 0.73940 | `baseline` | +0.00018 | 5/5 |
| `native_nan` | 0.73953 | `baseline` | +0.00032 | 4/5 |
| `typed_median` | 0.73937 | `median_flags0` | −0.00003 | 3/5 |
| `typed_nan` | 0.73956 | `native_nan` | +0.00002 | 2/5 |
| `nan_tuned` (nested Optuna) | 0.74000 | | | |
| `typed_tuned` | 0.74004 | `nan_tuned` | +0.00003 | 3/5 |
| `nan_catboost` | 0.74015 | | | |
| `typed_catboost` | 0.73992 | `nan_catboost` | −0.00024 | 1/5 |
| `nan_blend` (rank average) | **0.74052** | `baseline` | **+0.00130** | 5/5 |
| `typed_blend` | 0.74039 | `nan_blend` | −0.00014 | 0/5 |

## Interpretation

- **Treating flag blanks as 0 helped a little.** It raised AUC in all five folds, by
  +0.00018. With the median every row of these columns becomes 1; with 0, the few rows
  where 1 was actually recorded stand out. Why the gain is this small cannot be settled
  by this experiment alone, and that a blank means "no" is an assumption.
- **GAIN was not part of this comparison.** These results compare median, NaN and the
  type rules. They do not separate the effect of GAIN itself from the leakage of the
  competition pipeline (GAIN fitted before the split and with the target); that needs
  GAIN fitted inside each fold.
- **Separating the types did not help.** Once gaps are left as NaN, LightGBM learns inside
  each training fold where to send them, and the type is already recorded in other
  columns (`이식된 배아 수 = 0`, `동결 배아 사용 여부 = 0`, the treatment type). The
  explicit rules matched NaN within ±0.0003 for the fixed and tuned LightGBM, and were
  slightly worse for CatBoost and for the blend (lower in every fold).
- **The best result is also the simplest input.** Leaving gaps as NaN, with no
  hand-made ratios, and blending tuned LightGBM with CatBoost gave 0.74052, +0.0013 over
  the baseline in every fold. It is also +0.00019 above the blend in
  [IMPROVEMENT.md](IMPROVEMENT.md), which added ordinal counts and eleven ratio features
  (higher in four of five folds).

One seed and five folds that share training rows: differences are descriptive, not a
significance test. Cross-validation AUC on the training data is not comparable with
leaderboard scores. The recorded run's commit (`dirty: true`) predates this script's
commit; its `missing_types_cv_sha256` and `improve_cv_sha256` identify the executed
source, and copies of both files are kept with a hash manifest in the run's `source/`
folder.

## Reproduce

```bash
python -m pip install -r requirements-improve.txt
python missing_types_cv.py                             # about 45 minutes on a 15-core laptop
python missing_types_cv.py --sample 20000 --trials 3   # quick structural check
```
