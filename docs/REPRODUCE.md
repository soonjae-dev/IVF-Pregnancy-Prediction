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
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The test requirements include the baseline requirements and PyTorch 2.2.1 for
GAIN loss regression tests. The supported baseline itself does not use PyTorch.
The [GitHub Actions workflow](../.github/workflows/tests.yml) installs
`requirements-test.txt` on Ubuntu with Python 3.11 (CPU-only PyTorch wheels)
and runs this suite on every push and pull request.

The full `requirements.txt` is optional for the legacy training, tuning and study
scripts. Those scripts have validation limitations and are outside this supported
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
