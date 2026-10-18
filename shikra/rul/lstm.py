"""Attention-LSTM RUL estimation following Dida et al. (2025).

Implements a 2-layer LSTM with additive (Bahdanau-style) temporal attention
and a dense head, trained with causal sliding windows (window size 30) and
piecewise-linear RUL target capped at 125 cycles.

CPU-only, deterministic per seed via torch.manual_seed.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from shikra.rul.features import RUL_CAP, get_informative_sensors

WINDOW_SIZE = 30
DEFAULT_SEEDS = (2026, 2027, 2028)


# ---------------------------------------------------------------------------
# Attention-LSTM Model Architecture
# ---------------------------------------------------------------------------


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention over temporal sequence outputs."""

    def __init__(self, hidden_size: int = 64, attention_size: int = 64) -> None:
        super().__init__()
        self.w_a = nn.Linear(hidden_size, attention_size, bias=True)
        self.v_a = nn.Linear(attention_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Parameters
        ----------
        h : torch.Tensor of shape (batch_size, seq_len, hidden_size)

        Returns
        -------
        context : torch.Tensor of shape (batch_size, hidden_size)
        weights : torch.Tensor of shape (batch_size, seq_len, 1)
        """
        # score: (batch_size, seq_len, attention_size) -> (batch_size, seq_len, 1)
        score = self.v_a(torch.tanh(self.w_a(h)))
        weights = torch.softmax(score, dim=1)
        # context: sum over timesteps -> (batch_size, hidden_size)
        context = torch.sum(weights * h, dim=1)
        return context, weights


class AttentionLSTM(nn.Module):
    """Attention-LSTM network following Dida et al. 2025.

    Architecture:
      - 2 LSTM layers of 64 units
      - Additive attention mechanism over timesteps (window = 30)
      - Dense head mapping context vector (64) -> 32 -> 1
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dense_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_size=hidden_size, attention_size=hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_size)

        Returns
        -------
        torch.Tensor of shape (batch_size,) predicted RUL
        """
        out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        context, _ = self.attention(out)  # (batch_size, hidden_size)
        pred = self.head(context).squeeze(-1)  # (batch_size,)
        return pred


# ---------------------------------------------------------------------------
# Causal Window Construction
# ---------------------------------------------------------------------------


def create_causal_windows(
    df: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int = WINDOW_SIZE,
    is_test: bool = False,
    rul_cap: int = RUL_CAP,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build strictly causal sequence windows from C-MAPSS DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['subset', 'unit', 'cycle', sensors...].
    scaler : StandardScaler
        Scaler fit on training units only.
    window_size : int, default 30
        Length of sequence windows.
    is_test : bool, default False
        If True, only extracts the last window (ending at the final cycle)
        for each test unit.
    rul_cap : int, default 125
        Piecewise-linear RUL upper bound.

    Returns
    -------
    X : np.ndarray, shape (n_samples, window_size, n_features)
    y : np.ndarray, shape (n_samples,)
    meta : pd.DataFrame with ['subset', 'unit', 'cycle']
    """
    sensors = get_informative_sensors()
    windows_list: list[np.ndarray] = []
    targets_list: list[float] = []
    meta_list: list[dict[str, Any]] = []

    grouped = df.groupby(["subset", "unit"], sort=False)

    for (subset, unit), grp in grouped:
        grp_sorted = grp.sort_values("cycle").reset_index(drop=True)
        norm_data = scaler.transform(grp_sorted[sensors].values)
        n_cycles = len(grp_sorted)

        if is_test:
            # Official test sets: evaluate on the last cycle only
            if n_cycles >= window_size:
                w = norm_data[-window_size:]
            else:
                w = np.zeros((window_size, norm_data.shape[1]), dtype=np.float32)
                w[-n_cycles:] = norm_data

            windows_list.append(w.astype(np.float32))
            # Test target if available (rul_final)
            target = float(grp_sorted["rul_final"].iloc[-1]) if "rul_final" in grp_sorted.columns else 0.0
            targets_list.append(target)
            meta_list.append({
                "subset": subset,
                "unit": int(unit),
                "cycle": int(grp_sorted["cycle"].iloc[-1]),
            })
        else:
            # Training / validation trajectories: extract causal window at each cycle
            max_cycle = grp_sorted["cycle"].max()
            for idx in range(n_cycles):
                c = grp_sorted["cycle"].iloc[idx]
                if idx + 1 >= window_size:
                    w = norm_data[idx + 1 - window_size : idx + 1]
                else:
                    w = np.zeros((window_size, norm_data.shape[1]), dtype=np.float32)
                    w[-(idx + 1) :] = norm_data[: idx + 1]

                true_rul = float(max_cycle - c)
                capped_rul = float(min(true_rul, rul_cap))

                windows_list.append(w.astype(np.float32))
                targets_list.append(capped_rul)
                meta_list.append({
                    "subset": subset,
                    "unit": int(unit),
                    "cycle": int(c),
                })

    X = np.stack(windows_list, axis=0)
    y = np.array(targets_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    return X, y, meta


# ---------------------------------------------------------------------------
# Training and Prediction
# ---------------------------------------------------------------------------


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 2026,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
) -> AttentionLSTM:
    """Train an AttentionLSTM model with early stopping on validation MSE.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, 30, n_features)
    y_train : np.ndarray, shape (N,)
    X_val : np.ndarray, shape (M, 30, n_features)
    y_val : np.ndarray, shape (M,)
    seed : int
        RNG seed for torch and numpy.
    epochs : int, default 40
    batch_size : int, default 128
    lr : float, default 1e-3
    patience : int, default 10
        Early stopping patience on validation loss.

    Returns
    -------
    AttentionLSTM in eval mode with best validation weights restored.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[2]
    model = AttentionLSTM(input_size=n_features, hidden_size=64, num_layers=2)

    dataset_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataset_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_weights: dict[str, Any] | None = None
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for batch_x, batch_y in loader_train:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in loader_val:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_sum += loss.item()
                n_val_batches += 1

        val_mse = val_loss_sum / max(1, n_val_batches)

        if val_mse < best_val_loss - 1e-3:
            best_val_loss = val_mse
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    return model


def predict_lstm(model: AttentionLSTM, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run model inference on input windows.

    Parameters
    ----------
    model : AttentionLSTM
    X : np.ndarray of shape (N, 30, n_features)
    batch_size : int, default 256

    Returns
    -------
    np.ndarray of shape (N,) predicted RUL
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            p = model(batch_x)
            preds_list.append(p.cpu().numpy())

    return np.concatenate(preds_list, axis=0)
