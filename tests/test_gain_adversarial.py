import math
import unittest

import torch

from src.gain_imputer import generator_adversarial_loss


class TestGainAdversarial(unittest.TestCase):

    def test_loss_uses_only_missing_entries(self):
        probs = torch.tensor(
            [[0.2, 0.8], [0.4, 0.6]],
            requires_grad=True,
        )

        mask = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        loss = generator_adversarial_loss(probs, mask)

        # Only probabilities 0.8 and 0.4 matter.
        expected = -(math.log(0.8) + math.log(0.4)) / 2

        self.assertAlmostEqual(
            loss.item(), expected, places=5
        )

        loss.backward()

        expected_grad = torch.tensor([
            [0.0, -0.625],
            [-1.25, 0.0],
        ])

        torch.testing.assert_close(
            probs.grad, expected_grad
        )

    def test_observed_entries_do_not_affect_loss(self):
        mask = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        probs_a = torch.tensor([
            [0.2, 0.8],
            [0.4, 0.6],
        ])

        probs_b = torch.tensor([
            [0.9, 0.8],
            [0.4, 0.1],
        ])

        loss_a = generator_adversarial_loss(probs_a, mask)
        loss_b = generator_adversarial_loss(probs_b, mask)

        torch.testing.assert_close(loss_a, loss_b)

    def test_no_missing_entries(self):
        probs = torch.tensor(
            [[0.2, 0.8], [0.4, 0.6]],
            requires_grad=True,
        )

        mask = torch.ones_like(probs)

        loss = generator_adversarial_loss(probs, mask)

        self.assertEqual(loss.item(), 0.0)

        loss.backward()

        torch.testing.assert_close(
            probs.grad, torch.zeros_like(probs)
        )


if __name__ == "__main__":
    unittest.main()
