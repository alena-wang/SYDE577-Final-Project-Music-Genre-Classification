"""Tests for the Phase 3 experiment definitions and reporting."""

import contextlib
import importlib.util
import io
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("librosa")
                     and importlib.util.find_spec("matplotlib"), "install requirements.txt")
class ExperimentTests(unittest.TestCase):
    def test_configuration_parsing(self):
        from genre.experiments import parse_args

        defaults = parse_args([])
        self.assertEqual(len(defaults.configs), 8)
        self.assertEqual(tuple(defaults.seeds), (577, 578, 579, 580, 581))
        args = parse_args(["--configs", "deep-lr-0.01", "deep-dropout-0.2",
                           "--seeds", "578", "--epochs", "2", "--batch-size", "16",
                           "--patience", "1"])
        self.assertEqual([c.name for c in args.configs], ["deep-lr-0.01", "deep-dropout-0.2"])
        self.assertEqual([c.learning_rate for c in args.configs], [0.01, 0.001])
        self.assertEqual([c.dropout for c in args.configs], [0, 0.2])
        self.assertEqual((args.seeds, args.epochs, args.batch_size, args.patience), ([578], 2, 16, 1))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--seeds", "577", "577"])

    def test_parameter_counts(self):
        from genre.experiments import CONFIGURATIONS
        from genre.model import MLP

        expected = {(64,): 5258, (128, 64): 18122, (128, 64, 32): 19882, (243,): 19936}
        for config in CONFIGURATIONS:
            with self.subTest(config=config.name):
                model = MLP(config.hidden_sizes, config.dropout)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 expected[config.hidden_sizes])

    def test_dropout_behavior(self):
        import torch
        from torch import nn

        from genre.model import MLP

        torch.manual_seed(577)
        model = MLP((128, 64, 32), dropout=0.5)
        layers = [layer for layer in model.layers if isinstance(layer, nn.Dropout)]
        self.assertEqual([layer.p for layer in layers], [0.5, 0.5, 0.5])
        features = torch.ones(4, 71)
        with torch.inference_mode():
            self.assertFalse(torch.equal(model(features), model(features)))
            model.eval()
            torch.testing.assert_close(model(features), model(features), rtol=0, atol=0)
            for layer in layers:
                torch.testing.assert_close(layer(features), features, rtol=0, atol=0)

    def test_result_aggregation(self):
        from genre.experiments import aggregate

        settings = dict(hidden_sizes=[64], learning_rate=0.001, dropout=0, parameter_count=5258)
        records = [
            dict(settings, configuration=name, best_validation_accuracy=accuracy, accuracy_gap_pp=gap)
            for name, accuracy, gap in (("a", 0.6, 10), ("a", 0.8, 14), ("b", 0.65, -3))
        ]
        first, second = aggregate(records)
        self.assertEqual(first["runs"], 2)
        self.assertEqual(first["mean_best_validation_accuracy_pct"], 70)
        self.assertAlmostEqual(first["sd_best_validation_accuracy_pp"], 200**0.5)
        self.assertEqual(first["mean_accuracy_gap_pp"], 12)
        self.assertIsNone(second["sd_best_validation_accuracy_pp"])
