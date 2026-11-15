# The final model: train, save, load, predict

[`final_model.py`](../final_model.py) turns the best experiment
([SEEDS.md](experiments/SEEDS.md)) into a model that can be saved, loaded and used on
new rows.

- Training record: [`results/final_model_20261115T161344_326127Z.json`](../results/final_model_20261115T161344_326127Z.json).
- The model files themselves (`models/`) are git-ignored.

## What it is

| Part | Content |
|---|---|
| Input | The 67 competition columns, as read by `baseline_cv.load_dataset`. Numeric gaps stay NaN; string blanks become `__MISSING__`. No hand-made features. |
| LightGBM | String columns one-hot encoded (encoder fitted on the training rows). 25 Optuna trials on an inner 80/20 split with early stopping, then refit on all 256,351 rows. Chosen: `num_leaves` 39, `min_child_samples` 160, `colsample_bytree` 0.97, `subsample` 0.90, `reg_lambda` 1.35, 338 trees (inner AUC 0.7375). |
| CatBoost | Raw string categories. Tree count from early stopping on an inner split, then refit on all rows: 1,011 trees (inner AUC 0.7370). |
| Combine | Mean of the two predicted probabilities. |
| Training time | 280 s on a 15-core laptop (LightGBM 145 s, CatBoost 134 s). |

The fitting code repeats the settings of `improve_cv.lgb_tuned` and
`improve_cv.catboost_native`. A test fits both versions on the same synthetic data and
requires identical predictions, so the saved model is trained exactly the way the
experiments evaluated it.

## Combining by probability, not rank

The experiments combined the two models by averaging ranks. A rank depends on the other
rows being scored, so a patient's score would change with the batch, and a single row
has no rank at all. The final model averages probabilities instead. On the saved
out-of-fold predictions of the five split seeds:

| Split seed | Rank mean AUC | Probability mean AUC |
|---|---:|---:|
| 42 | 0.74052 | 0.74052 |
| 1 | 0.74037 | 0.74032 |
| 2 | 0.74018 | 0.74015 |
| 3 | 0.74032 | 0.74029 |
| 4 | 0.74027 | 0.74024 |

The difference is at most 0.00005. The probability mean also averages 0.258, the same as
the positive rate. On split seed 42, the predicted and observed pregnancy rates agree
within 0.003 in every decile of the score. That is calibration measured on
cross-validation predictions. It has not been checked on data from another period or
clinic.

## Output and what it means

`predict` writes `ID, probability`. The probability is the model's estimate of the chance
of pregnancy for one treatment cycle, given **the full record of that cycle**. That
record includes inputs known only after treatment has progressed, such as the number of
embryos transferred. It is not a pre-treatment prediction
([ERROR_ANALYSIS.md](experiments/ERROR_ANALYSIS.md)). A row's score does not depend on
the other rows in the file; a test checks that one row scored alone gets the same value.

## Input rules

| Situation | Behaviour |
|---|---|
| A training column is missing | Error, nothing is scored |
| A column the model does not use (for example `ID`) | Ignored and listed in the report |
| A numeric column has a value that is not a number | Error, with the column and count |
| Numbers read as text, blanks | Converted; a blank becomes missing |
| A string category never seen in training | Scored and counted in the report. LightGBM sees no category switched on; CatBoost treats it as a new category |
| The target column is present | Error |
| `ID` duplicated or blank | Error |

On the competition test file (90,067 rows) the only unseen values were 3 rows of
`특정 시술 유형`.

## Save and load

- The bundle (`model.joblib`) holds:
  - the column list and which columns are categorical;
  - the categories seen in training;
  - the encoder and both models;
  - the training details.
- `manifest.json` records:
  - the bundle's SHA-256;
  - the versions of numpy, pandas, scikit-learn, lightgbm, catboost, optuna and joblib;
  - the data and source hashes.
- Loading checks the bundle hash first and refuses a file that does not match. A joblib
  bundle is a pickle, and loading a pickle runs code, so load only bundles you built.
- Loading warns when the installed package versions differ from the ones used in training.

## Verification

| Check | Result |
|---|---|
| Test file scored before saving and after loading the saved bundle | 90,067 rows, identical (largest difference 0.0) |
| `predict` command on the test file, in a new process | Same file hash as the predictions written at training time (`7ed2801e…`) |
| Synthetic tests (`tests/test_final_model.py`, 8 tests) | Same predictions as the experiment code, save/load round trip, tampered bundle refused, version mismatch warns, scores within [0, 1], one row alone scores the same as in a batch, input rules |

The test predictions average 0.2586 (range 0.0004 to 0.75). No leaderboard score exists
for them: the competition is closed.

## Commands

```bash
python -m pip install -r requirements-improve.txt
python final_model.py train                     # about 5 minutes; also scores data/test.csv
python final_model.py predict --model models/final_<time> --input data/test.csv
python final_model.py train --sample 5000 --trials 2   # quick check
```

The tests in `tests/test_final_model.py` are skipped when CatBoost or Optuna is not
installed. CI installs both, and `tests/test_pipeline_e2e.py` also runs `train` and
`predict` as commands on synthetic data.
