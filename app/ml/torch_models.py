"""CPU-only PyTorch models: an MLP regressor and a GRU forecaster.

Training loops accept an ``epoch_callback`` so the WebSocket routes can stream
per-epoch loss (CLAUDE.md: websockets for NN epochs). Epoch counts are hard
capped by settings.max_epochs (<=50)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from app.config import get_settings

EpochCallback = Optional[Callable[[int, float], None]]

torch.manual_seed(42)


class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_layers: int = 2, neurons: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(max(1, hidden_layers)):
            layers += [nn.Linear(prev, neurons), nn.ReLU()]
            prev = neurons
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers=max(1, num_layers), batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def _clamp_epochs(epochs: int) -> int:
    return max(1, min(int(epochs), get_settings().max_epochs))


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameters: dict,
    epoch_callback: EpochCallback = None,
) -> MLPRegressor:
    epochs = _clamp_epochs(hyperparameters.get("epochs", 100))
    lr = float(hyperparameters.get("learning_rate", 1e-3))
    hidden_layers = int(hyperparameters.get("hidden_layers", 2))
    neurons = int(hyperparameters.get("neurons", 64))

    model = MLPRegressor(x_train.shape[1], hidden_layers, neurons)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if epoch_callback is not None:
            epoch_callback(epoch, float(loss.item()))
    model.eval()
    return model


def predict_mlp(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).view(-1).numpy()


def _windows(series: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        xs.append(series[i : i + seq_len])
        ys.append(series[i + seq_len])
    x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(ys), dtype=torch.float32).view(-1, 1)
    return x, y


def train_gru(
    series: np.ndarray,
    hyperparameters: dict,
    epoch_callback: EpochCallback = None,
) -> tuple[GRUForecaster, float, float, int]:
    """Train a GRU on a 1D series. Returns (model, mean, std, seq_len); mean/std
    are the normalization stats needed to invert predictions."""
    epochs = _clamp_epochs(hyperparameters.get("epochs", 50))
    lr = float(hyperparameters.get("learning_rate", 1e-3))
    hidden_size = int(hyperparameters.get("hidden_size", 64))
    num_layers = int(hyperparameters.get("num_layers", 2))
    seq_len = max(2, int(hyperparameters.get("sequence_length", 12)))

    mean, std = float(series.mean()), float(series.std() or 1.0)
    norm = (series - mean) / std
    x, y = _windows(norm, seq_len)

    model = GRUForecaster(hidden_size, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if epoch_callback is not None:
            epoch_callback(epoch, float(loss.item()))
    model.eval()
    return model, mean, std, seq_len
