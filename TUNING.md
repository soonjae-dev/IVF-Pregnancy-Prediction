# Re-tuning the hyperparameters

The parameters in `configs/*.json` were chosen by a search that applied SMOTE to
the whole training set before cross-validating, which is why their `best_value`
fields read around 0.89 against an honest out-of-fold figure of about 0.73. The
protocol was fixed; the parameters it picked were never re-picked. This is how
to replace them.

It runs on a laptop over days, not in one sitting, so everything here is
resumable.

---

## The shape of it

```
tune.py              search, on 75% of the data, 3-fold, resumable
validate_tuning.py   score the winners against the incumbents on the other 25%
ensemble_study.py    re-measure the published figures, once something is adopted
```

The 75/25 split is the point. A search returns the best of everything it tried,
and that maximum is inflated by however many things it tried — so it cannot be
compared against anything. The 25% referee split was used to choose nothing, so
a win there is a win. `tune.py` never writes to `configs/`; only
`validate_tuning.py --write` does, and only for models that won.

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

It builds `data/gain_cache.npz` if it is missing, and also gives you the current
published figures to compare against later. Delete that file if you ever change
the feature pipeline, or every later step will quietly measure the old
representation.

The build happens in a separate process, on purpose. GAIN is the only step that
needs PyTorch, and every study script imports LightGBM, XGBoost and CatBoost at
module level. On macOS each of those links its own OpenMP runtime, and loading
PyTorch's on top of them takes the process down with a bare
`zsh: segmentation fault` before GAIN prints anything. You can see it for
yourself:

```bash
python -c "import torch; x=torch.randn(2000,200); print((x@x.T).sum())"                      # fine
python -c "import lightgbm, xgboost, catboost; import torch; x=torch.randn(2000,200); print((x@x.T).sum())"   # segfault
```

Both orders crash, so there is no safe import order to find — the two have to
stay in separate processes. `build_cache.py` runs GAIN in an interpreter that
has never imported a boosting library, and `src/optimization.py` holds no torch
at all (the TabTransformer, which needs it, lives in `src/tab_transformer.py`),
so `tune.py` and the study scripts never load it either.

`main.py` is the exception and is still affected: it needs GAIN and the four
models in one process. Use the study scripts and `tune.py` on macOS.

## 2. Search, one model at a time

```bash
caffeinate -i python tune.py --model lgb --trials 60
```

`caffeinate -i` keeps the machine awake while it runs; without it a closed lid
ends the run. (The study survives, but the trials stop.)

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
time. `lgb` and `xgb` take a trial count; 200 is a few minutes and worth it,
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

It writes nothing yet. Adopt what won:

```bash
python validate_tuning.py --write
```

Each replaced file is kept as `configs/<model>_params.leaky.json`, so the
originals stay in the repository as the record of what the leaky protocol chose.

## 4. Re-measure

`configs/` has changed, so every figure in the README is now stale:

```bash
rm -f data/gain_cache.npz     # only if the feature pipeline changed too
python ensemble_study.py
python resampling_study.py    # slower; the SMOTE table
```

Then update the README's Results tables from the output, and delete the **Known
limitation** paragraph about parameters selected under the leaky protocol —
that caveat is what this whole exercise removes.

---

## Notes

**Nothing tune.py prints belongs in the README.** Its scores are 3-fold ROC-AUC
on 75% of the data, chosen as the maximum over every trial. They exist to rank
parameter sets against each other. The reportable numbers come from
`ensemble_study.py`, over five folds, on everything.

**Do not change `SPLIT_SEED`.** It is the same constant in both scripts and it
is what keeps the referee split unseen. Changing it after tuning has started
hands the referee data the search has already looked at, and the comparison
stops meaning anything.

**Residual optimism does not go to zero.** The referee split settles *which*
parameters to use. The README's figures still come from folds drawn over all the
data, including the 75% the search saw, so they keep a little optimism. Removing
that entirely takes nested cross-validation — tuning inside each outer fold —
which is five searches instead of one. Worth knowing; not worth the week.

**If a run dies partway**, nothing is lost. Trials are committed to SQLite as
they complete. Re-run the same command.

**If the fans never stop and everything crawls**, the machine is thermally
throttled. `--hours` with a smaller number, run more often, is kinder to a
fanless laptop than one long pin at 100%.
