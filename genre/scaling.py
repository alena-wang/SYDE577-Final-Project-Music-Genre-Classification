"""Compute a reusable feature transform from training rows."""

import numpy as np


def standardization(train_features):
    mean = train_features.mean(axis=0, dtype=np.float64)
    scale = train_features.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1
    return mean, scale
