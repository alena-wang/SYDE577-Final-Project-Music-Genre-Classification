"""Phase 1 checks, including a real librosa extraction from generated audio."""

import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from genre.dataset import GENRES, KNOWN_GROUPS, LABELS, assign_splits, duplicate_groups


class DatasetTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(LABELS, dict(blues=0, classical=1, country=2, disco=3,
                                     hiphop=4, jazz=5, metal=6, pop=7, reggae=8, rock=9))

    def test_split_integrity(self):
        hashes = {f"{genre}.{index:05d}": f"{genre}-{index}"
                  for genre in GENRES for index in range(100)}
        hashes["blues.00000"] = hashes["pop.00037"]
        groups = duplicate_groups(hashes)
        assignments = assign_splits(groups)
        self.assertEqual(assignments, assign_splits(dict(reversed(list(groups.items())))))
        self.assertEqual(set(assignments), set(hashes))
        for group in KNOWN_GROUPS:
            self.assertEqual(len({assignments[recording] for recording in group}), 1)
        self.assertEqual(assignments["blues.00000"], assignments["pop.00015"])
        for genre in GENRES:
            counts = Counter(split for key, split in assignments.items()
                             if key.startswith(genre + "."))
            for split, target in (("train", 70), ("validation", 15), ("test", 15)):
                self.assertLessEqual(abs(counts[split] - target), 2)


@unittest.skipUnless(importlib.util.find_spec("librosa"), "install requirements.txt for audio tests")
class FeatureTests(unittest.TestCase):
    def test_feature_dimensions(self):
        import numpy as np
        import soundfile

        from genre.features import FEATURE_NAMES, SAMPLE_RATE, extract

        t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
        with tempfile.TemporaryDirectory(dir=".") as folder:
            path = Path(folder) / "tone.wav"
            soundfile.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), SAMPLE_RATE)
            values = extract(path)
        self.assertEqual(len(set(FEATURE_NAMES)), 71)
        self.assertEqual(values.shape, (71,))
        self.assertTrue(np.isfinite(values).all())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "install requirements.txt for scaling tests")
class ScalingTests(unittest.TestCase):
    def test_training_standardization(self):
        import numpy as np

        from genre.scaling import standardization

        mean, scale = standardization(np.array([[1, 3], [5, 3]]))
        np.testing.assert_array_equal(mean, [3, 3])
        np.testing.assert_array_equal(scale, [2, 1])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("librosa"),
                     "install requirements.txt for model tests")
class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        import torch

        from genre.model import ShallowClassifier

        model = ShallowClassifier()
        features = torch.zeros(4, 71)
        self.assertEqual(tuple(model(features).shape), (4, 10))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5258)
        torch.testing.assert_close(model.predict_proba(features).sum(dim=1), torch.ones(4))

    def test_deep_output_shape_and_parameter_count(self):
        import torch

        from genre.model import MLP

        model = MLP()
        self.assertEqual(tuple(model(torch.zeros(4, 71)).shape), (4, 10))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 19882)
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in model.layers))

    def test_deep_training_updates_parameters_but_validation_does_not(self):
        import math

        import torch

        from genre.model import MLP
        from genre.train import run_epoch

        torch.manual_seed(577)
        model = MLP(hidden_sizes=(16, 8), dropout=0.2)
        batches = [(torch.randn(10, 71), torch.arange(10))]
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        initial = [parameter.detach().clone() for parameter in model.parameters()]

        loss, accuracy = run_epoch(model, batches, loss_function, optimizer)
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertTrue(model.training)
        fitted = [parameter.detach().clone() for parameter in model.parameters()]
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(initial, fitted)))

        run_epoch(model, batches, loss_function)
        self.assertFalse(model.training)
        for expected, actual in zip(fitted, model.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
