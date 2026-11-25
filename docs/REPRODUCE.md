# Reproduce the supported baseline

Run commands from the repository root with Python 3.11. The recorded local
baseline used Python 3.11.18, NumPy 1.26.4, pandas 2.2.1, scikit-learn
1.4.1.post1 and LightGBM 4.3.0. These are the pinned baseline dependencies;
matching them does not guarantee identical results on every platform.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
```

`requirements-baseline.txt` is sufficient for `baseline_cv.py`, `analyze_oof.py`
and `compare_missingness.py`. If importing LightGBM on macOS fails because
`libomp` is missing, install the OpenMP runtime with `brew install libomp`, then
retry the import. This system package step is conditional on that error.

To run the complete synthetic-data test suite, also install the test requirements:

```bash
python -m pip install -r requirements-test.txt -r requirements-improve.txt
python -m unittest discover -s tests -v
```

The test requirements include the baseline requirements and PyTorch 2.2.1 for
GAIN loss regression tests. The supported baseline itself does not use PyTorch.
The [GitHub Actions workflow](../.github/workflows/tests.yml) installs
`requirements-test.txt` and `requirements-improve.txt` on Ubuntu with Python 3.11
(CPU-only PyTorch wheels) and runs this suite on every push and pull request.

`tests/test_pipeline_e2e.py` runs `improve_cv.py`, `missing_types_cv.py`,
`seed_cv.py`, `error_analysis.py` and `final_model.py train`/`predict` as commands on
synthetic files with the competition's schema (`tests/synthetic_ivf.py`: 1,500 training
and 300 test rows, random values, structural gaps as in the real data). Tuning is cut
to 2 trials. It checks wiring, not scores: OOF rows line up with the input, labels are
intact, scores are finite, recorded hashes and metrics match the saved files, split
seed 42 reproduces `missing_types_cv.py`, and the final model's predictions survive a
save and reload. Without CatBoost and Optuna these tests are skipped.

The full `legacy/requirements.txt` is optional for the legacy training, tuning and study
scripts. Install it in a separate virtual environment rather than on top of the baseline
environment: it adds XGBoost, imbalanced-learn, skorch and a pinned PyTorch, and without
them `main.py`, `evaluate_cv.py` and the `*_study.py` scripts fail at import. The legacy
scripts live in `legacy/` and are run from inside that folder (`cd legacy`); they read
the data from `../data/`.

```bash
uv venv --python 3.11 .venv-legacy        # or: python3.11 -m venv .venv-legacy
uv pip install --python .venv-legacy/bin/python -r legacy/requirements.txt
```

`.venv-legacy/` is ignored by Git. On 2026-11-15 this environment imported every legacy
script, passed the synthetic-data test suite, and was used to check the cache path end to
end on a 2,000-row extract in a temporary directory: the first request built the cache,
identical requests reused it, and changing the GAIN iterations, the encoded columns or the
training CSV each kept the old file as `.stale-<timestamp>.npz` and rebuilt it.
`evaluate_cv.py` built the cache on its first run and reused it on the second with
identical results. No full-data legacy run was repeated. Those scripts have validation limitations and are outside this supported
baseline workflow. Installing their dependencies does not correct their design.
See the [archived notebook guide](archive/NOTEBOOK.md) for the historical notebook.

## Supply data

Obtain the competition training CSV separately and place it at `data/train.csv`.
Raw data is not distributed with this repository. This workflow needs the
training CSV only; `test.csv` is not required for cross-validation. The baseline
requires the target column `임신 성공 여부` to contain only nonmissing 0 or 1
values, validates those values before integer conversion, and removes that column and
`ID` from the input features. Use the same authorized data version when comparing
results, and retain the dataset hash recorded by each run.

The synthetic-data tests run without competition data. They validate code
behavior; they do not reproduce the reported full-data metrics.

## Train and save OOF predictions

Start with a stratified sample, then run all rows:

```bash
python baseline_cv.py --sample 5000 --folds 3
python baseline_cv.py --folds 5
```

| Option | Default | Meaning |
|---|---|---|
| `--data` | `data/train.csv` | Input training CSV, relative to the working directory unless absolute. |
| `--sample` | All rows | Number of rows to retain by stratified sampling with seed 42. |
| `--folds` | `5` | Shuffled stratified CV folds; each class must have enough rows for every fold. |
| `--estimators` | `250` | Number of LightGBM estimators. |
| `--output-dir` | `results` | Parent directory for a new timestamped run directory. |

The seed is fixed at 42 in the code; there is no `--seed` option. The baseline
fits numeric median imputation, categorical one-hot encoding and LightGBM within
each training fold. It performs no GAIN, resampling, supervised feature selection
or hyperparameter search.

Each run creates `baseline_<UTC timestamp>/` under the output parent and prints
the saved paths. `--output-dir` chooses the parent, not the final run name.
Existing runs are preserved. A run contains:

- `summary.json`: pooled OOF metrics, fold metrics, model parameters, data and
  source hashes, versions, and available Git revision/working-tree state.
- `predictions_oof.csv`: `row_index`, `source_row_index`, `fold`, `y_true` and
  `oof_probability`.

`row_index` identifies positions within the evaluated sample. `source_row_index`
is the zero-based data-row position in the input CSV, including after sampling;
it is not a patient identifier. Raw data and prediction CSVs remain local and
are excluded from Git. The source file hash and recorded Git state describe the
execution; the recorded commit alone is insufficient when the tree was modified.
Outside a Git repository, the revision and working-tree state can be unavailable.

## Inspect a run without retraining

Replace the quoted directory below with the directory printed by your own run:

```bash
IVF_RUN_DIR="results/baseline_REPLACE_WITH_YOUR_RUN"
python analyze_oof.py "$IVF_RUN_DIR/predictions_oof.csv" --summary "$IVF_RUN_DIR/summary.json"
```

This validates saved row positions, labels, probabilities and fold IDs; with a
summary it checks row/fold counts and metrics, and verifies the recorded OOF file
hash when present. It prints a JSON analysis. Add `--output path/to/new_report.json`
to save a report; its parent directory must already exist and an existing file
is never overwritten. Prediction CSVs are not included in the public repository,
so published summary paths alone are insufficient to run this command.

ROC-AUC is the primary competition metric. The best F1 threshold printed by this
analysis is selected and scored on the same OOF labels and is exploratory.
Validate a frozen threshold independently before treating that F1 as an estimate
of future performance.

## Predict the competition test set

```bash
python predict_test.py
```

This fits the same fixed baseline (no tuning, seed 42) on all training rows and
predicts `data/test.csv`, which must be obtained the same way as the training
file. It creates `results/submission_<UTC timestamp>/` with `submission.csv`
(`ID`, `probability`, in test-file order) and `summary.json` (data, test and
source hashes). The CSV stays local. Options: `--data`, `--test`,
`--estimators`, `--output-dir`.

## Improvement and missing-value-type experiments

```bash
python -m pip install -r requirements-improve.txt
python improve_cv.py
python missing_types_cv.py
```

Both reuse the baseline's folds and record source hashes. `improve_cv.py --variants`
accepts a subset in any order and always runs it in ladder order; `blend` requires
`lgb_tuned` and `catboost`. See [IMPROVEMENT.md](experiments/IMPROVEMENT.md) and
[MISSING_TYPES.md](experiments/MISSING_TYPES.md).

The subgroup analysis needs no training. It reads the OOF files of the latest
`missing_types_cv.py` run and refuses to run if their hashes or the training data differ
from what that run recorded:

```bash
python error_analysis.py
```

See [ERROR_ANALYSIS.md](experiments/ERROR_ANALYSIS.md).

The four key models on five split seeds (about 1h40 on a 15-core laptop):

```bash
python seed_cv.py
python seed_cv.py --seeds 42,1 --sample 5000 --trials 2   # quick check
```

See [SEEDS.md](experiments/SEEDS.md).

Train, save and use the final model (about 5 minutes):

```bash
python final_model.py train
python final_model.py predict --model models/final_<time> --input data/test.csv
```

See [FINAL_MODEL.md](FINAL_MODEL.md).

## Repeat the controlled missingness comparison

```bash
python compare_missingness.py
```

This compares numeric median imputation with and without missingness indicators
on the same rows, folds and fixed model settings: all rows, five stratified folds,
seed 42 and 250 estimators. Its only CLI options are `--data` and `--output-dir`,
with the same defaults as the baseline. It creates a fresh
`missingness_<UTC timestamp>/` directory containing `comparison.json`,
`baseline_oof.csv` and `missing_indicators_oof.csv`. The JSON retains paired fold
and pooled OOF differences, timing, settings and provenance; the CSVs stay local.
The one-seed difference is descriptive and does not establish a reliable gain.

For the reported results and their validation limits, return to the
[project README](../README.md).
