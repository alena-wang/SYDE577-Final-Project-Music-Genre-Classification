"""Fully connected classifiers for fixed-length audio features."""

import torch
from torch import nn

from .dataset import GENRES
from .features import FEATURE_COUNT

SHALLOW_WIDTHS = (64,)
DEEP_WIDTHS = (128, 64, 32)


class MLP(nn.Module):
    def __init__(self, hidden_sizes=DEEP_WIDTHS, dropout=0.0):
        super().__init__()
        if not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError("Provide at least one positive hidden width.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")
        layers = []
        input_size = FEATURE_COUNT
        for width in hidden_sizes:
            layers.extend((nn.Linear(input_size, width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = width
        layers.append(nn.Linear(input_size, len(GENRES)))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)

    def predict_proba(self, features):
        return torch.softmax(self(features), dim=-1)


class ShallowClassifier(MLP):
    def __init__(self):
        super().__init__(SHALLOW_WIDTHS)
