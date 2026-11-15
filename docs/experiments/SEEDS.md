# Does the gain hold on other splits?

The best result (`nan_blend`, OOF ROC-AUC 0.74052, +0.0013 over the baseline) came from
one split: `StratifiedKFold(5, shuffle=True, random_state=42)`. This experiment repeats
the four models that matter on five split seeds.

- Script: [`seed_cv.py`](../../seed_cv.py).
- Result: [`results/seeds_20261115T142328_492960Z/comparison.json`](../../results/seeds_20261115T142328_492960Z/comparison.json).
  It records the data and source SHA-256 hashes and package versions: numpy 1.26.4,
  pandas 2.2.1, scikit-learn 1.4.1.post1, lightgbm 4.3.0, catboost 1.2.5, optuna 3.6.1.
- Models, all on the same input as `missing_types_cv.py`'s `native_nan`: numeric gaps
  left as NaN and no hand-made features. See [MISSING_TYPES.md](MISSING_TYPES.md).

  | Model | Setup |
  |---|---|
  | `baseline` | `baseline_cv.py` unchanged |
  | `nan_tuned` | LightGBM; 25 Optuna trials on an inner 80/20 split of each training fold |
  | `nan_catboost` | CatBoost on raw categories; tree count from an inner split |
  | `nan_blend` | Equal-weight rank average of the two |

- **Only the split seed changes** (42, 1, 2, 3, 4, chosen before the run). Model and tuning
  randomness stays at seed 42: LightGBM, CatBoost, the Optuna sampler and the inner splits.
  The spread below is sensitivity to the split alone. Varying the model seeds as well
  would add their variation and was not done.
- Within one split seed all four models see the same rows and folds, so the differences
  are paired.
- **Check**: split seed 42 reproduced the earlier `missing_types` run exactly. The AUC
  difference is 0 for all four models.

## Results

OOF ROC-AUC by split seed:

| Split seed | baseline | nan_tuned | nan_catboost | nan_blend | blend − baseline | Folds improved | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.73922 | 0.74000 | 0.74015 | **0.74052** | +0.00130 | 5/5 | 21.7 min |
| 1 | 0.73907 | 0.73997 | 0.73992 | **0.74037** | +0.00130 | 5/5 | 18.9 min |
| 2 | 0.73889 | 0.73965 | 0.73983 | **0.74018** | +0.00129 | 5/5 | 20.5 min |
| 3 | 0.73922 | 0.74001 | 0.73982 | **0.74032** | +0.00109 | 5/5 | 19.1 min |
| 4 | 0.73900 | 0.73994 | 0.73974 | **0.74027** | +0.00127 | 5/5 | 20.2 min |

Across the five seeds:

| Model | AUC mean (SD) | AUC range | Gain over baseline: mean (SD), range | Folds improved | AP mean | Time per seed |
|---|---|---|---|---:|---:|---:|
| baseline | 0.73908 (0.00014) | 0.73889–0.73922 | — | — | 0.44831 | 15 s |
| nan_tuned | 0.73991 (0.00015) | 0.73965–0.74001 | +0.00083 (0.00008), +0.00076 to +0.00093 | 25/25 | 0.44980 | 10.9 min |
| nan_catboost | 0.73989 (0.00016) | 0.73974–0.74015 | +0.00081 (0.00014), +0.00060 to +0.00094 | 25/25 | 0.44968 | 8.9 min |
| nan_blend | 0.74033 (0.00013) | 0.74018–0.74052 | +0.00125 (0.00009), +0.00109 to +0.00130 | 25/25 | 0.45068 | ~0 s (uses the two) |

AP rose with AUC on every seed (blend − baseline +0.0018 to +0.0029).

## What this shows

1. **The gain does not depend on the split.** Tuned LightGBM, CatBoost and the blend each
   beat the baseline on every split seed and in every one of the 25 folds. The blend's
   gain varies between seeds by an SD of 0.00009, much smaller than the gain itself.
2. **Blending helps on every split.** The blend is above both of its parts on every seed,
   by 0.0003 to 0.0004 over the better of the two.
3. **0.74052 was the best of the five splits.** The mean is 0.74033. Quoting 0.7405 alone
   slightly overstates the model; "about 0.7403, +0.0012 to +0.0013 over the baseline" is
   the fairer summary.
4. **The split moves the baseline as much as the gain is uncertain.** The baseline ranges
   0.73889–0.73922 (0.0003 wide). That is why the gains are compared within a seed on
   the same folds, not across seeds.
5. **Neither single model is consistently ahead of the other.** CatBoost is ahead on
   seeds 42 and 2, tuned LightGBM on 1, 3 and 4. Combining them is what is consistent.
6. **Cost**: about 20 minutes per seed on a 15-core laptop for a gain of about 0.0012 AUC.
   Almost all of it is the LightGBM tuning (11 min) and CatBoost (9 min); the baseline
   takes 15 seconds.

## What it does not show

- The five splits reuse the same 256,351 rows. Agreement across them shows the gain is
  not an artefact of one split. They are not independent samples, and they do not
  replace a held-out test set. Nothing here says how the model would score on the
  competition's test set.
- Model and tuning randomness was fixed. A different Optuna or CatBoost seed could move
  each model a little further.
- The subgroup caveats of [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) still apply, for
  example that much of the 0.74 comes from rows with no embryo transferred.

## Reproduce

```bash
python seed_cv.py                                        # about 1h40
python seed_cv.py --seeds 42,1 --sample 5000 --trials 2  # quick check
```

The OOF predictions for each seed are written next to `comparison.json` and are
git-ignored.
