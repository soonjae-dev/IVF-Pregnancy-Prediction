# Controlled comparison: improving the fold-fitted baseline

## Question and fixed protocol

After the leak was removed, the baseline scored 0.7392 OOF ROC-AUC. Which changes
raise that score, and is missing-value handling the main lever?

[`improve_cv.py`](../../improve_cv.py) runs a cumulative ladder of variants. Every
variant uses all 256,351 rows and the same five shuffled stratified folds (seed 42) as
`baseline_cv.py`, so fold-by-fold differences are paired. Everything learned from data
is learned inside the training part of each outer fold:

| Step | Change from the previous step | Learned inside the training fold |
|---|---|---|
| `baseline` | `baseline_cv.py` unchanged: median imputation, one-hot, LightGBM (250 trees) | Medians, category list |
| `native_nan` | Numeric gaps left as NaN; LightGBM learns which side missing values go | Category list |
| `di_zero` | Gaps meaning "not applicable" set to 0 on DI rows: numeric columns that are empty for every DI row of the training fold (28 columns in each fold) | Column list, category list |
| `ordinal` | Ten count columns (`0회` … `6회 이상`) as numbers 0–6; patient and donor age bands also added as ordered numbers | Category list |
| `features` | Eleven row-wise ratios (embryos transferred or stored per embryo, embryos per egg, fertilisation rate, prior pregnancy rate and others); no label is used | Category list |
| `lgb_tuned` | LightGBM tuned with Optuna (25 trials) on an inner 80/20 split of the training fold, tree count by early stopping on that split, then refitted on the whole training fold | Everything above plus hyperparameters |
| `catboost` | CatBoost on the same features with native categorical handling; tree count by early stopping on an inner split, then refitted | Category statistics, tree count |
| `blend` | Equal-weight average of the ranks of `lgb_tuned` and `catboost` | Nothing (no weights are fitted) |

The validation fold is never used for imputation, column choice, tuning, early
stopping or blending weights.

## Results

[Recorded run](../../results/improve_20261115T064857_957864Z/comparison.json)

| Step | OOF ROC-AUC | Average precision | vs previous step (pooled) | Folds up vs previous | vs baseline (pooled) |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.73922 | 0.4487 | | | |
| `native_nan` | 0.73953 | 0.4494 | +0.00032 | 4/5 | +0.00032 |
| `di_zero` | 0.73952 | 0.4492 | −0.00002 | 2/5 | +0.00030 |
| `ordinal` | 0.73935 | 0.4489 | −0.00017 | 1/5 | +0.00013 |
| `features` | 0.73900 | 0.4485 | −0.00035 | 0/5 | −0.00022 |
| `lgb_tuned` | 0.74002 | 0.4501 | +0.00102 | 5/5 | +0.00080 |
| `catboost` | 0.73981 | 0.4498 | −0.00021 | 1/5 | +0.00059 |
| `blend` | **0.74034** | 0.4509 | +0.00053 | 5/5 | **+0.00112** |

The `baseline` step reproduces the reference run exactly (0.7392190176), which
confirms that every step used the same folds. `catboost` is compared with
`lgb_tuned` in the "previous step" column; both use the same features.

## Interpretation

- **Missing-value handling moved the score by at most +0.0003.** Leaving gaps as NaN
  helped slightly; setting DI gaps to 0 added nothing once the model could already see
  NaN and the treatment type. Together with the indicator experiment
  ([MISSINGNESS.md](MISSINGNESS.md), +0.00015), this does not support the idea that
  imputation is what separates the baseline from the top of the leaderboard.
- **The hand-made ratios did not help** (−0.00035, lower in all five folds). Tree models
  can already combine the raw counts, and the ratios add noise.
- **Tuning and model diversity helped most.** Nested tuning added +0.0010 in every fold,
  and the rank blend of LightGBM and CatBoost added another +0.0005 in every fold. The
  final +0.0011 over the baseline is small but consistent across folds.
- **This is not a leaderboard score.** 0.7403 is cross-validation AUC on the training
  data. It cannot be compared with the 1st-place public score (0.74350) or private score
  (0.74377), which come from a different, hidden test set.

One seed and five folds that share training rows: the fold standard deviations in the
JSON are descriptive, not confidence intervals or significance tests. The recorded run's
Git commit is the one before `improve_cv.py` was committed (`dirty: true`); its
`improve_cv_sha256` identifies the executed source.

## Not yet tested

- GAIN fitted inside each training fold, compared with these steps on the same folds.
- More seeds or repeated folds, to put a spread on the small differences above.
- Group or time splits, which need patient or clinic identifiers the data does not have.

## Reproduce

```bash
python -m pip install -r requirements-improve.txt
python improve_cv.py                      # about 17 minutes on a 15-core laptop
python improve_cv.py --sample 20000 --trials 3   # quick structural check
```

Each run writes a new `results/improve_<UTC timestamp>/` with `comparison.json` and one
OOF CSV per step. The CSVs stay local and are excluded from Git.
