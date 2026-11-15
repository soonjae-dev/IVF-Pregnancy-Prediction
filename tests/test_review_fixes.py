"""Regression tests for the issues raised in the 2026-11-15 project review."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import build_cache
from improve_cv import VARIANTS, select_variants


class TestCacheSpec(unittest.TestCase):
    """A cache is reused only when data, code and build options all match."""

    def setUp(self):
        self.previous = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        pathlib.Path("data").mkdir()
        pathlib.Path("configs").mkdir()
        pathlib.Path("data/train.csv").write_text("a\n1\n")
        pathlib.Path("data/test.csv").write_text("a\n2\n")
        pathlib.Path("configs/config.json").write_text(json.dumps({
            "pipeline": {"train_data_path": "data/train.csv",
                         "test_data_path": "data/test.csv"},
            "imputation": {"gain_iterations": 1000}}))
        self.cache = pathlib.Path("data/cache.npz")

    def tearDown(self):
        os.chdir(self.previous)
        self.tmp.cleanup()

    def write_cache(self, spec):
        np.savez_compressed(self.cache, X=np.zeros((1, 1)), y=np.zeros(1), columns=np.array(["a"]),
                            X_test=np.zeros((1, 1)), test_ids=np.array(["t"]),
                            spec=np.array(json.dumps(spec)))

    def ensure(self, **options):
        with mock.patch.object(build_cache.subprocess, "run") as run:
            build_cache.ensure_cache(self.cache, quiet=True, **options)
        return run.called

    def stale_files(self):
        return list(pathlib.Path("data").glob("cache.stale-*.npz"))

    def test_matching_cache_is_reused(self):
        self.write_cache(build_cache.cache_spec())
        self.assertFalse(self.ensure())
        self.assertTrue(self.cache.exists())

    def test_default_iterations_and_explicit_value_match(self):
        self.write_cache(build_cache.cache_spec())
        self.assertFalse(self.ensure(iterations=1000))

    def test_changed_options_trigger_rebuild(self):
        self.write_cache(build_cache.cache_spec())
        self.assertTrue(self.ensure(iterations=10))
        self.assertEqual(len(self.stale_files()), 1, "the old cache is kept, not deleted")

    def test_changed_encode_list_triggers_rebuild(self):
        self.write_cache(build_cache.cache_spec())
        self.assertTrue(self.ensure(encode=["시술 유형_원본"]))

    def test_changed_data_triggers_rebuild(self):
        self.write_cache(build_cache.cache_spec())
        pathlib.Path("data/train.csv").write_text("a\n3\n")
        self.assertTrue(self.ensure())

    def test_cache_without_record_triggers_rebuild(self):
        np.savez_compressed(self.cache, X=np.zeros((1, 1)), y=np.zeros(1), columns=np.array(["a"]),
                            X_test=np.zeros((1, 1)), test_ids=np.array(["t"]))
        self.assertTrue(self.ensure())

    def test_spec_covers_preprocessing_sources(self):
        sources = build_cache.cache_spec()["sources"]
        for name in ["src/gain_imputer.py", "src/encoding.py", "src/feature_selection.py"]:
            self.assertIn(name, sources)


class TestVariantOrder(unittest.TestCase):

    def test_any_order_runs_in_ladder_order(self):
        self.assertEqual(select_variants("baseline,blend,lgb_tuned,catboost"),
                         ["baseline", "lgb_tuned", "catboost", "blend"])
        self.assertEqual(select_variants(",".join(reversed(VARIANTS))), VARIANTS)

    def test_invalid_selections_fail_before_training(self):
        for text in ["native_nan", "baseline,blend,lgb_tuned", "baseline,unknown",
                     "baseline,baseline"]:
            with self.assertRaises(ValueError, msg=text):
                select_variants(text)


GAIN_CHECK = """
import contextlib, io, json
import numpy as np
import torch
from src import gain_imputer

modes = []
original = gain_imputer.Generator.forward
def recording_forward(self, x):
    modes.append([self.training, torch.is_grad_enabled()])
    return original(self, x)
gain_imputer.Generator.forward = recording_forward

rng = np.random.default_rng(0)
data = rng.uniform(0, 1, size=(64, 4)).astype(np.float32)
data[rng.uniform(size=data.shape) < 0.2] = -1.0
with contextlib.redirect_stdout(io.StringIO()):
    imputed = gain_imputer.gain_impute(data, batch_size=16, iterations=3,
                                       device=torch.device("cpu"), seed=0)
observed = ~np.isclose(data, -1.0)
print(json.dumps({"first": modes[0], "last": modes[-1],
                  "observed_kept": bool(np.allclose(imputed[observed], data[observed], atol=1e-6)),
                  "finite": bool(np.isfinite(imputed).all())}))
"""


class TestGainInference(unittest.TestCase):

    def test_final_imputation_runs_in_eval_mode(self):
        # GAIN runs in its own interpreter, as in build_cache.py: this test
        # process has already imported LightGBM, and torch alongside it can
        # crash on macOS (see docs/archive/TUNING.md).
        root = pathlib.Path(__file__).resolve().parents[1]
        completed = subprocess.run([sys.executable, "-c", GAIN_CHECK], cwd=root,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["last"], [False, False], "final pass: eval mode, no gradients")
        self.assertTrue(result["first"][0], "training passes stay in training mode")
        self.assertTrue(result["observed_kept"])
        self.assertTrue(result["finite"])


if __name__ == "__main__":
    unittest.main()
