#!/usr/bin/env python3
"""
tune.py — re-run the hyperparameter search under a protocol that is not leaking.

Every `best_value` in configs/*.json came from a search that applied SMOTE to
the whole training set before cross-validation, which is why they read around
0.89 while the honest out-of-fold figure is around 0.73. The parameters
themselves were selected under that protocol and have been carried forward
unchanged ever since. This replaces them.

Protocol
--------
The data is split once, 75/25 stratified, with a fixed seed:

    tuning split (75%)   everything Optuna is allowed to see
    referee split (25%)  never touched here; validate_tuning.py uses it to
                         decide whether the new parameters actually beat the
                         incumbents, on data neither of them was chosen on

Inside the tuning split, each trial is scored by 3-fold ROC-AUC. Three folds
rather than the RepeatedStratifiedKFold(5, n_repeats=2) that main.py uses,
because that is ten fits per trial and this has to finish. The search is for
ranking parameter sets, not for producing a number to report -- no figure from
this script belongs in the README.

Trials report each fold as it completes, so a hopeless set is pruned after one
or two fits instead of three.

Running it
----------
The study lives in a SQLite file, so the search survives being interrupted.
Stop it with Ctrl-C, close the laptop, run the same command tomorrow: it picks
up with the trials it already has.

    python tune.py --model lgb --trials 60
    python tune.py --model rf  --hours 8
    python tune.py --status

One model at a time is deliberate. Each model already uses every core, so
running two at once makes both slower, not the pair faster.
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
SPLIT_SEED = 20250913        # the 75/25 split; changing it invalidates the referee
TUNING_CV_SEED = 1337        # inner folds, deliberately not the evaluation seed
CACHE = pathlib.Path("data/gain_cache.npz")
STUDY_DB = pathlib.Path("tuning/optuna.db")
RESULTS = pathlib.Path("tuning/best_params")
MODELS = ["xgb", "lgb", "rf", "cat"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_matrix():
    """The imputed matrix the studies cache. Built by any of them."""
    if not CACHE.exists():
        raise SystemExit(
            f"{CACHE} not found.\n"
            "Build it once with:  python ensemble_study.py\n"
            "(or resampling_study.py / column_study.py -- they share the cache)")
    cached = np.load(CACHE, allow_pickle=True)
    return cached["X"], cached["y"]


def tuning_split(X, y):
    """The 75% Optuna may see. The other 25% is the referee's."""
    X_tune, _, y_tune, _ = train_test_split(
        X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
    return X_tune, y_tune


def objective(trial, model, X, y, folds):
    from src.optimization import build_model, suggest_params

    params = suggest_params(trial, model)
    cv = StratifiedKFold(folds, shuffle=True, random_state=TUNING_CV_SEED)
    scores = []

    for step, (fit_index, valid_index) in enumerate(cv.split(X, y)):
        estimator = build_model(model, params, seed=SEED)
        estimator.fit(X[fit_index], y[fit_index])
        proba = estimator.predict_proba(X[valid_index])[:, 1]
        scores.append(roc_auc_score(y[valid_index], proba))

        # Report as we go so a bad set dies after one fold, not three.
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run(model, trials, hours, folds):
    X, y = load_matrix()
    X_tune, y_tune = tuning_split(X, y)
    log(f"{model}: tuning on {len(y_tune):,} rows x {X.shape[1]} features "
        f"({len(y) - len(y_tune):,} held back for the referee)")

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=model,
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        log(f"resuming: {done} completed trial(s) already in {STUDY_DB}, "
            f"best so far {study.best_value:.4f}")

    def callback(study_, trial_):
        if trial_.state == optuna.trial.TrialState.COMPLETE:
            mark = "  <- best" if trial_.number == study_.best_trial.number else ""
            log(f"  trial {trial_.number:3d}  {trial_.value:.4f}{mark}")
        elif trial_.state == optuna.trial.TrialState.PRUNED:
            log(f"  trial {trial_.number:3d}  pruned")
        save_best(study_, model)

    try:
        study.optimize(
            lambda trial: objective(trial, model, X_tune, y_tune, folds),
            n_trials=trials,
            timeout=int(hours * 3600) if hours else None,
            callbacks=[callback],
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        log("interrupted — the study is saved, rerun the same command to continue")

    save_best(study, model)
    log(f"{model}: best {study.best_value:.4f} over "
        f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} "
        f"completed trial(s)")
    log(f"  {json.dumps(study.best_params, ensure_ascii=False)}")


def save_best(study, model):
    """
    Write the leader to tuning/best_params/, never to configs/.

    configs/ is what the pipeline runs on. Nothing lands there until
    validate_tuning.py has scored these parameters against the incumbents on
    the referee split.
    """
    try:
        best = dict(study.best_params)
    except ValueError:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    best["tuning_cv_auc"] = float(study.best_value)
    best["_note"] = ("3-fold ROC-AUC on the 75% tuning split. A search score, "
                     "not a performance estimate. Not for the README.")
    with open(RESULTS / f"{model}_params.json", "w") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2)


def status():
    if not STUDY_DB.exists():
        log(f"no study yet at {STUDY_DB}")
        return
    print(f"  {'model':6s}{'done':>7s}{'pruned':>8s}{'best':>10s}")
    for model in MODELS:
        try:
            study = optuna.load_study(study_name=model,
                                      storage=f"sqlite:///{STUDY_DB}")
        except KeyError:
            print(f"  {model:6s}{'-':>7s}{'-':>8s}{'-':>10s}")
            continue
        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED]
        best = f"{study.best_value:.4f}" if complete else "-"
        print(f"  {model:6s}{len(complete):>7d}{len(pruned):>8d}{best:>10s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS,
                        help="which model to tune; one at a time")
    parser.add_argument("--trials", type=int, default=50,
                        help="trials to add in this run (default 50)")
    parser.add_argument("--hours", type=float, default=None,
                        help="stop after this many hours, whatever the trial count")
    parser.add_argument("--folds", type=int, default=3,
                        help="inner CV folds per trial (default 3)")
    parser.add_argument("--status", action="store_true",
                        help="show trial counts and current leaders, then exit")
    args = parser.parse_args()

    if args.status:
        status()
        return
    if not args.model:
        parser.error("--model is required (or use --status)")

    started = time.time()
    run(args.model, args.trials, args.hours, args.folds)
    log(f"elapsed {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
