#!/usr/bin/env python3
"""
build_cache.py — run steps 1-6 of main.py once and cache the result.

Every study script needs the same thing to start: the matrix that comes out of
preprocessing, feature engineering, correlation-based selection, one-hot
encoding and GAIN imputation. GAIN takes minutes and the comparisons do not
depend on re-running it, so it is built once and cached.

Three of those scripts used to carry their own copy of this code, and the copies
had already drifted -- one wrote the cache without the `columns` key another one
reads. There is one copy now.

Running it in its own process is the other half of the point. GAIN is the only
part of the pipeline that needs PyTorch, and the study scripts import LightGBM,
XGBoost and CatBoost at module level. On macOS each of those links its own
OpenMP runtime, and in the original local macOS environment loading PyTorch's on
top of them took the process down with a segmentation fault before GAIN printed
anything (not verified on other installations). Keeping torch in a
process that has never imported a boosting library sidesteps that entirely, and
costs one subprocess spawn.

So other scripts call ensure_cache(), which re-executes this file rather than
importing it. Importing this module does NOT import torch.

Usage:
  python build_cache.py                          # the pipeline's default columns
  python build_cache.py --force                  # rebuild even if cached
  python build_cache.py --out ../data/other.npz --encode "시술 유형_원본" --encode "시술 당시 나이"
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

TARGET = "임신 성공 여부"
SEED = 42
DEFAULT_CACHE = pathlib.Path("../data/gain_cache.npz")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# Keys every cache carries. A cache written before X_test was added is missing
# some of these, so it is rebuilt rather than half-read.
REQUIRED_KEYS = {"X", "y", "columns", "X_test", "test_ids", "spec"}
ROOT = pathlib.Path(__file__).resolve().parent
CONFIG = pathlib.Path("configs/config.json")
# Every source file whose behaviour shapes the cached matrix.
SOURCES = ["build_cache.py", "src/config.py", "src/preprocessing.py", "src/features.py",
           "src/feature_selection.py", "src/encoding.py", "src/run_imputation.py",
           "src/gain_imputer.py"]


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_spec(encode=None, iterations=None, config_path=CONFIG):
    """
    Everything that determines the cached matrix: input data, the code that
    transforms it, and the build options. A cache is reused only when its stored
    spec equals the spec of the request.
    """
    config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    pipeline = config["pipeline"]
    return {
        "train_sha256": _sha256(pipeline["train_data_path"]),
        "test_sha256": _sha256(pipeline["test_data_path"]),
        "config_sha256": _sha256(config_path),
        "sources": {name: _sha256(ROOT / name) for name in SOURCES},
        "encode": list(encode) if encode else "default",
        "iterations": int(iterations or config["imputation"]["gain_iterations"]),
        "seed": SEED,
    }


def stored_spec(path):
    """The spec a cache was built with, or None if it is unreadable or incomplete."""
    try:
        with np.load(path, allow_pickle=True) as cached:
            if not REQUIRED_KEYS.issubset(set(cached.files)):
                return None
            return json.loads(str(cached["spec"]))
    except Exception:
        return None


def _set_aside(path):
    """
    Keep a stale cache under a new name instead of deleting it.

    The name carries microseconds and a counter, and the move never replaces an
    existing file: os.link fails if the target exists (rename would overwrite it
    silently on POSIX), so two set-asides in the same instant both survive.
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    for counter in range(1000):
        suffix = "" if counter == 0 else f"-{counter}"
        stale = path.with_name(f"{path.stem}.stale-{stamp}{suffix}{path.suffix}")
        try:
            os.link(path, stale)
        except FileExistsError:
            continue
        path.unlink()
        return stale
    raise RuntimeError(f"could not find a free name to keep {path}")


def ensure_cache(out=DEFAULT_CACHE, encode=None, iterations=None, quiet=False):
    """
    Return the cache path, building it in a separate process if it is missing
    or was built from different data, code or options.

    Callers get one command for the user and an isolated interpreter for torch.
    Raises CalledProcessError if the build fails, so a crash is not mistaken for
    an empty cache.
    """
    out = pathlib.Path(out)
    if out.exists():
        wanted = cache_spec(encode, iterations)
        found = stored_spec(out)
        if found == wanted:
            if not quiet:
                log(f"using cached matrix {out}")
            return out
        stale = _set_aside(out)
        if not quiet:
            reason = "has no build record" if found is None else "was built differently"
            log(f"{out} {reason}; kept as {stale.name} and rebuilding")

    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--out", str(out)]
    for column in encode or []:
        command += ["--encode", column]
    if iterations:
        command += ["--iterations", str(iterations)]

    if not quiet:
        log(f"{out} missing — building it in a separate process")
    subprocess.run(command, check=True)
    return out


def build(out, encode, iterations, force):
    out = pathlib.Path(out)
    if out.exists() and not force:
        log(f"{out} already exists; pass --force to rebuild")
        return

    # Imported here, not at module scope, so that `from build_cache import
    # ensure_cache` stays free of torch. See the module docstring.
    import torch

    from src.config import get_device, set_seed
    from src.encoding import run_encoding_pipeline
    from src.feature_selection import run_feature_selection
    from src.features import run_feature_engineering
    from src.preprocessing import run_basic_preprocessing
    from src.run_imputation import run_gain_on_dataframes

    config = json.load(open(CONFIG))
    iterations = iterations or config["imputation"]["gain_iterations"]
    # Recorded before any work, so the stored spec describes the inputs used.
    spec = cache_spec(encode, iterations)
    set_seed(SEED)
    device = get_device()

    log("preprocessing, feature engineering, selection, encoding")
    train_df, test_df, test_ids = run_basic_preprocessing(
        config["pipeline"]["train_data_path"], config["pipeline"]["test_data_path"])
    train_df, test_df = run_feature_engineering(train_df, test_df)
    train_df, test_df = run_feature_selection(train_df, test_df)
    if encode:
        train_df, test_df = run_encoding_pipeline(train_df, test_df,
                                                  columns_to_encode=list(encode))
    else:
        train_df, test_df = run_encoding_pipeline(train_df, test_df)

    log(f"GAIN imputation ({iterations} iterations) on {device}")
    started = time.time()
    train_df, test_df = run_gain_on_dataframes(train_df, test_df, device,
                                               iterations=iterations)
    log(f"  GAIN finished in {time.time() - started:.0f}s")

    y = train_df[TARGET].astype(int).values
    features = train_df.drop(columns=[TARGET])
    X = features.values.astype(np.float32)

    # The test matrix rides along so that main.py can produce submissions
    # without ever importing torch. synchronize_columns has already aligned the
    # two frames, so one column list describes both.
    if list(test_df.columns) != list(features.columns):
        raise RuntimeError("train and test columns diverged after imputation")
    X_test = test_df.values.astype(np.float32)

    if cache_spec(encode, iterations) != spec:
        raise RuntimeError("input data or source changed during the build; nothing saved")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=y, X_test=X_test,
        columns=np.array(features.columns, dtype=object),
        test_ids=np.array([] if test_ids is None else test_ids.values, dtype=object),
        spec=np.array(json.dumps(spec, sort_keys=True)))
    log(f"cached {X.shape[0]:,} train and {X_test.shape[0]:,} test rows "
        f"x {X.shape[1]} features to {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_CACHE))
    parser.add_argument("--encode", action="append", default=None,
                        help="override columns_to_encode; repeat per column")
    parser.add_argument("--iterations", type=int, default=None,
                        help="GAIN iterations (default: configs/config.json)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the cache exists")
    args = parser.parse_args()
    build(args.out, args.encode, args.iterations, args.force)


if __name__ == "__main__":
    main()
