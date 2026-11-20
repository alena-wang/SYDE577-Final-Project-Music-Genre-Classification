"""Focused tests for frozen checkpoint evaluation and confusion metrics."""

import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(module)
                         for module in ("torch", "librosa", "matplotlib")), "install requirements.txt")
class EvaluationTests(unittest.TestCase):
    def test_eval_uses_frozen_weights_and_disables_dropout(self):
        import torch

        from genre.evaluate import evaluate
        from genre.model import MLP

        torch.manual_seed(577)
        model = MLP((8,), dropout=0.75)
        features = torch.randn(11, 71)
        labels = torch.arange(11) % 10
        snapshots = [p.detach().clone() for p in model.parameters()]

        def verify_mode(module, inputs):
            self.assertFalse(module.training)
            self.assertFalse(torch.is_grad_enabled())

        hook = model.register_forward_pre_hook(verify_mode)
        loss, matrix = evaluate(model, features, labels, batch_size=4)
        hook.remove()
        expected = torch.nn.functional.cross_entropy(model(features), labels)
        self.assertAlmostEqual(loss, expected.item(), places=6)
        self.assertEqual(int(matrix.sum()), 11)
        for parameter, snapshot in zip(model.parameters(), snapshots):
            torch.testing.assert_close(parameter, snapshot, rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    def test_macro_scores_include_zero_for_missing_predictions(self):
        import numpy as np

        from genre.evaluate import classification_scores

        scores, (precision, recall, f1) = classification_scores(np.array([[2, 0], [1, 0]]))
        np.testing.assert_allclose(precision, [2 / 3, 0])
        np.testing.assert_allclose(recall, [1, 0])
        np.testing.assert_allclose(f1, [0.8, 0])
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertAlmostEqual(scores["macro_precision"], 1 / 3)
        self.assertAlmostEqual(scores["macro_recall"], 0.5)
        self.assertAlmostEqual(scores["macro_f1"], 0.4)
