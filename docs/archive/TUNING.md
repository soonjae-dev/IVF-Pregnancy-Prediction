> **Archived runbook — validation limitations (2026-11-11).** The commands below
> document the legacy shared-cache workflow, not the supported baseline. Target-based
> feature selection and GAIN imputation occur before its tuning/referee split.
> The split does not make the full pipeline leakage-free. Use the
> [current README](../../README.md) for supported reproduction and read the
> [validation audit](../VALIDATION_AUDIT.md) before interpreting legacy results.
> All commands below assume the repository root as the working directory.

# Archived hyperparameter tuning workflow

The original search applied SMOTE to the whole training set before
cross-validation. Its inflated `best_value` fields remain in
`configs/*_params.leaky.json`. The current XGBoost, LightGBM, RandomForest and
CatBoost parameter files have already been replaced by the legacy re-tuning
workflow described here and record `referee_auc` instead. They are historical
configurations, not the parameters used by `baseline_cv.py`.

Moving SMOTE inside validation folds corrected one error, but the shared-cache
preprocessing still uses information from validation rows. The historical
search and referee scores below do not establish uncontaminated performance.

It runs on a laptop over days, not in one sitting, so everything here is
resumable.

---

## The shape of it

```
tune.py              search, on 75% of the data, 3-fold, resumable
validate_tuning.py   score the winners against the incumbents on the other 25%
ensemble_study.py    measure historical shared-cache variants after a config change
```

The 75/25 split separates hyperparameter search from the referee comparison at
the estimator stage. The maximum search score is selected over many trials and
is not a performance estimate. The shared cache was nevertheless learned before
this split, and the referee was later used for adoption decisions, so it is not
an independent final test. `tune.py` never writes to `configs/`; only
`validate_tuning.py --write` does, for models with a positive observed AUC delta.
The default validation command still overwrites `tuning/referee.json`.

---

## Setup

In VS Code: **Terminal → New Terminal**, from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then **Cmd+Shift+P → Python: Select Interpreter** and pick `./.venv/bin/python`,
so the editor and the terminal agree.

On Apple Silicon, LightGBM and XGBoost need OpenMP, which macOS does not ship:

```bash
brew install libomp
```

Check the install with `python -c "import lightgbm, xgboost, catboost, optuna"`.
Silence there means it worked.

Put `train.csv` and `test.csv` from Dacon in `data/`.

---

## 1. Build the imputed matrix once

Every script reads the same cached matrix, so pay for GAIN once:

```bash
python ensemble_study.py
```

It builds `data/gain_cache.npz` if it is missing and reports legacy shared-cache
metrics. If the feature pipeline changes, preserve any cache needed to interpret
previous runs and rebuild the default cache with `python build_cache.py --force`.
Cache reuse checks required keys, not source or data hashes, so an existing cache
can otherwise silently retain an older representation.

The build happens in a separate process, on purpose. GAIN is the only step that
needs PyTorch, and every study script imports LightGBM, XGBoost and CatBoost at
module level. In the historical local macOS environment, mixing the boosting
libraries with PyTorch caused a bare `zsh: segmentation fault` before GAIN printed
anything.
The following commands were used to investigate that environment-specific issue;
the failure is not asserted for every macOS installation:

```bash
python -c "import torch; x=torch.randn(2000,200); print((x@x.T).sum())"                      # fine
python -c "import lightgbm, xgboost, catboost; import torch; x=torch.randn(2000,200); print((x@x.T).sum())"   # segfault
```

Both import orders crashed in that local environment, so the legacy workflow
isolates the libraries in separate processes. `build_cache.py` runs GAIN in an
interpreter that has never imported a boosting library, and `src/optimization.py`
holds no torch
at all (the TabTransformer, which needs it, lives in `src/tab_transformer.py`),
so `tune.py` and the study scripts never load it either.

`main.py` needs both too, and gets them the same way: it reads the cache rather
than calling GAIN itself, so it holds no torch either.

## 2. Search, one model at a time

```bash
caffeinate -i python tune.py --model lgb --trials 60
```

`caffeinate -i` inhibits idle sleep on macOS while the command runs. It does not
guarantee execution through lid closure or other system sleep conditions. Trials
completed before an interruption remain in the study database.

One model at a time is deliberate. Each already uses every core, so two at once
makes both slower rather than the pair faster.

Stop any time with **Ctrl-C**. The study lives in `tuning/optuna.db`; the same
command tomorrow continues from the trials it already has:

```bash
python tune.py --status
```

```
  model    done  pruned      best
  xgb        18       7    0.7288
  lgb        60      22    0.7371
  rf          -       -         -
  cat         -       -         -
```

Measured on an M-series MacBook Air, on the full 104-feature matrix with 3
folds, counting pruned trials in the average:

| model | per trial | 100 trials |
|---|---|---|
| `lgb` | ~7 s | ~12 min |
| `rf` | ~2.4 min | ~4 h |
| `cat` | ~2.4 min | ~4 h |

`xgb` was not timed on its own; it sits between `lgb` and the other two. Expect
a fanless machine to slow down as it heats up.

Pruning is doing real work in those numbers: a hopeless parameter set dies after
one fold instead of three, and roughly 40% of trials get pruned once the median
pruner has enough history to judge by.

The gap between `lgb` and the rest is large enough to change how you spend the
time. `lgb` and `xgb` take a trial count; 200 is roughly 25 minutes for `lgb` and worth it,
since 6 hyperparameters is not a space 60 trials covers. `rf` and `cat` are
better given a clock:

```bash
caffeinate -i python tune.py --model lgb --trials 200
caffeinate -i python tune.py --model cat --hours 6
```

`--trials` is how many to add *this run*, not a target. Giving `lgb` 200 when it
already has 60 leaves it with 260, not 200.

## 3. Referee

Once every model has trials — 40 or more each is a reasonable place to stop:

```bash
python validate_tuning.py
```

```
  model            incumbent     tuned     Δ AUC      Δ F1   verdict
  LightGBM            0.7336    0.7404   +0.0068   +0.0049   adopt
  ...
```

Without `--write`, this command leaves `configs/` unchanged but writes or
overwrites `tuning/referee.json`. Preserve the earlier decision report before
rerunning if it is needed for traceability. The historical adoption command is:

```bash
python validate_tuning.py --write
```

The first replaced file is kept as `configs/<model>_params.leaky.json` if that
backup does not already exist. Subsequent replacements do not create additional
backups. These files record what the original protocol chose; adoption by this
workflow does not validate the shared-cache preprocessing.

## 4. Re-measure

Changing legacy `configs/` affects the study scripts, not the current baseline
with its fixed model settings. Historical follow-up commands were:

```bash
python ensemble_study.py
python resampling_study.py    # slower; the historical SMOTE table
```

If preprocessing changed, preserve the earlier artifacts and explicitly rebuild
the cache as described above. Label any new scores as legacy shared-cache
experiments and retain the validation limitations. They must not replace the
current README baseline results without a new fold-fitted evaluation.

---

## Notes

**Search scores are not performance estimates.** `tune.py` reports 3-fold
ROC-AUC on 75% of the shared matrix, with the best score selected over trials.
`ensemble_study.py` uses five folds over that matrix but retains the shared-cache
limitations. Both belong in the historical record, not the current baseline
results table.

**Keep `SPLIT_SEED` consistent when inspecting this workflow.** It is the same
constant in both scripts and preserves their estimator-stage partition. Changing
it after tuning starts can place search rows in the referee partition. Keeping
it fixed does not repair preprocessing learned before the split.

**Independent evaluation remains unresolved.** The referee split was used to
select parameters, and subsequent full-data CV includes rows seen by the search.
A new evaluation must fit every learned preprocessing step within each outer
training fold and put supervised selection and tuning inside inner folds. Nested
validation alone cannot repair a shared matrix learned from all rows in advance.

**If a run dies partway**, nothing is lost. Trials are committed to SQLite as
they complete. Re-run the same command.

**If trials keep getting slower and the machine runs hot**, it is likely thermally
throttled. `--hours` with a smaller number, run more often, is kinder to a
fanless laptop than one long pin at 100%.
