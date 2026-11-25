
import unittest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

from src.gain_imputer import reconstruction_loss


class TestGainReconstruction(unittest.TestCase):

    def test_reconstruction_loss_and_gradient(self):
        observed = torch.tensor([
            [2.0, 3.0],
            [5.0, 7.0],
        ])

        mask = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        generated = torch.zeros_like(
            observed, requires_grad=True
        )

        loss = reconstruction_loss(
            generated, observed, mask
        )

        self.assertAlmostEqual(
            loss.item(), 13.25
        )

        loss.backward()

        expected_gradient = torch.tensor([
            [-1.0, 0.0],
            [0.0, -3.5],
        ])

        torch.testing.assert_close(
            generated.grad, expected_gradient
        )


if __name__ == "__main__":
    unittest.main()
