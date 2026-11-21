"""Model definitions for emotion recognition (Valence & Arousal).

Models:
1. RidgeRegressionBaseline: Separate RidgeCV models for Valence and Arousal.
2. DeterministicMLP: 2-layer MLP (Linear -> LayerNorm -> GELU -> Dropout) predicting [mu_v, mu_a].
3. HeteroscedasticMLP: Same backbone outputting [mu_v, mu_a, raw_sigma_v, raw_sigma_a],
   smoothly bounding sigma to [0.05, 1.0] via 0.05 + 0.95 * sigmoid(raw_sigma).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV


class RidgeBaseline:
    """Baseline using independent RidgeCV models for Valence and Arousal."""

    def __init__(self, alphas: Optional[np.ndarray] = None):
        if alphas is None:
            alphas = np.logspace(-3, 5, 50)
        self.alphas = alphas
        self.model_v = RidgeCV(alphas=self.alphas)
        self.model_a = RidgeCV(alphas=self.alphas)
        self.sigma_v_train: float = 0.1
        self.sigma_a_train: float = 0.1

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeBaseline:
        """Fit separate RidgeCV models.

        Args:
            X: feature matrix of shape (N, D)
            y: target array of shape (N, 2) where y[:, 0] is valence and y[:, 1] is arousal
        """
        self.model_v.fit(X, y[:, 0])
        self.model_a.fit(X, y[:, 1])

        # Estimate residual standard deviations on training data
        preds_v = self.model_v.predict(X)
        preds_a = self.model_a.predict(X)
        self.sigma_v_train = float(np.std(y[:, 0] - preds_v))
        self.sigma_a_train = float(np.std(y[:, 1] - preds_a))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mu and constant residual sigma.

        Returns:
            mu: shape (N, 2)
            sigma: shape (N, 2)
        """
        mu_v = self.model_v.predict(X)
        mu_a = self.model_a.predict(X)
        mu = np.column_stack([mu_v, mu_a])

        sigma_v = np.full_like(mu_v, fill_value=max(0.05, self.sigma_v_train))
        sigma_a = np.full_like(mu_a, fill_value=max(0.05, self.sigma_a_train))
        sigma = np.column_stack([sigma_v, sigma_a])
        return mu, sigma


class DeterministicMLP(nn.Module):
    """2-layer deterministic MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear head."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting [mu_v, mu_a]. Shape: (batch_size, 2)."""
        h = self.backbone(x)
        return self.head(h)


class HeteroscedasticMLP(nn.Module):
    """Heteroscedastic MLP predicting mean and uncertainty for Valence and Arousal.

    Architecture:
        Backbone: Linear -> LayerNorm -> GELU -> Dropout
        Output Head: Linear(hidden_dim, 4) producing:
            - mu_v, mu_a (unbounded means)
            - raw_sigma_v, raw_sigma_a (smoothly mapped to [0.05, 1.0] via 0.05 + 0.95 * sigmoid)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        sigma_min: float = 0.05,
        sigma_max: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_range = sigma_max - sigma_min

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 4)

        # Initialize head weights: small random weights, biases near 0
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            mu: Tensor of shape (batch_size, 2) representing [mu_v, mu_a]
            sigma: Tensor of shape (batch_size, 2) strictly in [0.05, 1.0]
        """
        h = self.backbone(x)
        out = self.head(h)
        mu = out[:, :2]
        raw_sigma = out[:, 2:]
        sigma = self.sigma_min + self.sigma_range * torch.sigmoid(raw_sigma)
        return mu, sigma


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss for independent dimensions.

    Loss per sample and dimension:
        0.5 * ( log(sigma^2) + (target - mu)^2 / sigma^2 )
        = log(sigma) + 0.5 * ((target - mu) / sigma)^2

    Args:
        mu: Tensor of shape (batch_size, D)
        sigma: Tensor of shape (batch_size, D)
        target: Tensor of shape (batch_size, D)
        reduction: 'mean', 'sum', or 'none'
    """
    variance = torch.clamp(sigma**2, min=1e-6)
    loss = 0.5 * (torch.log(variance) + (target - mu) ** 2 / variance)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss
