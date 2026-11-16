"""Fail when competition data, private material or model weights are tracked by git."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Paths that must never enter the public repository.
FORBIDDEN = [
    (re.compile(r"^data/"), "competition data folder"),
    (re.compile(r"^models/"), "saved model folder"),
    (re.compile(r"^docs/private/"), "private application material"),
    (re.compile(r"^competition_archive/"), "original competition working folder"),
    (re.compile(r"\.(csv|parquet|npz|pt|cbm|pkl|joblib)$", re.IGNORECASE),
     "data file or model weights"),
]


def tracked_files():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        return None
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            capture_output=True, check=True)
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


class TestRepoHygiene(unittest.TestCase):
    def test_no_data_private_or_weights_tracked(self):
        files = tracked_files()
        if files is None:
            self.skipTest("not a git checkout")
        offending = [f"{path} ({reason})" for path in files
                     for pattern, reason in FORBIDDEN if pattern.search(path)]
        self.assertEqual(offending, [], "Files that must stay out of git are tracked")


if __name__ == "__main__":
    unittest.main()
