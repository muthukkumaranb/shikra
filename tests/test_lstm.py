"""Unit tests for the T1.4 Attention-LSTM RUL model."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from shikra.rul.features import get_informative_sensors  # noqa: E402
from shikra.rul.lstm import (  # noqa: E402
    AdditiveAttention,
    AttentionLSTM,
    create_causal_windows,
)


def test_model_output_shape() -> None:
    """Test AttentionLSTM model and AdditiveAttention output tensor shapes."""
    batch_size = 8
    seq_len = 30
    n_features = 14
    hidden_size = 64

    # 1. Test AdditiveAttention module
    attention = AdditiveAttention(hidden_size=hidden_size, attention_size=hidden_size)
    h = torch.randn(batch_size, seq_len, hidden_size)
    context, weights = attention(h)

    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len, 1)
    # Weights should sum to 1.0 along the sequence dimension
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size, 1), atol=1e-5)

    # 2. Test full AttentionLSTM model
    model = AttentionLSTM(input_size=n_features, hidden_size=hidden_size, num_layers=2)
    x = torch.randn(batch_size, seq_len, n_features)
    pred = model(x)

    assert pred.shape == (batch_size,)


def test_causal_windowing() -> None:
    """Test that sequence windows are strictly causal and contain no future data."""
    sensors = get_informative_sensors()
    n_cycles = 50

    # Synthetic trajectory where values strictly equal the cycle number
    data = {"subset": ["FD001"] * n_cycles, "unit": [1] * n_cycles, "cycle": list(range(1, n_cycles + 1))}
    for s in sensors:
        data[s] = [float(c) for c in range(1, n_cycles + 1)]

    df = pd.DataFrame(data)

    scaler = StandardScaler()
    scaler.fit(df[sensors].values)

    window_size = 30
    X, y, meta = create_causal_windows(df, scaler, window_size=window_size, is_test=False, rul_cap=125)

    assert len(X) == n_cycles
    assert len(y) == n_cycles
    assert len(meta) == n_cycles

    # For every window corresponding to cycle c:
    # 1. The last element in the window must correspond to normalized values at cycle c.
    # 2. Values in the window must only reflect cycles <= c (never > c).
    for idx in range(n_cycles):
        c = int(meta["cycle"].iloc[idx])
        w = X[idx]  # shape (30, n_features)

        # Unscale the last timestep of the window
        last_timestep_unscaled = scaler.inverse_transform(w[-1:])
        expected_val = float(c)
        assert np.allclose(last_timestep_unscaled, expected_val, atol=1e-4), (
            f"At cycle {c}, window tail did not match cycle value"
        )

        # In unscaled values, no element should exceed cycle c
        all_timesteps_unscaled = scaler.inverse_transform(w)
        # Non-padded rows must have values <= c
        if idx + 1 < window_size:
            # Padded prefix: the first (window_size - idx - 1) rows were zero-padded in scaled space
            real_rows = all_timesteps_unscaled[window_size - (idx + 1) :]
            assert np.all(real_rows <= expected_val + 1e-4)
        else:
            assert np.all(all_timesteps_unscaled <= expected_val + 1e-4)


def test_determinism_with_fixed_seed() -> None:
    """Test that identical seeds yield identical model weights and predictions."""
    input_size = 14
    seq_len = 30
    x = torch.randn(4, seq_len, input_size)

    # Run 1
    torch.manual_seed(2026)
    model1 = AttentionLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    model1.eval()
    with torch.no_grad():
        out1 = model1(x)

    # Run 2 with same seed
    torch.manual_seed(2026)
    model2 = AttentionLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    model2.eval()
    with torch.no_grad():
        out2 = model2(x)

    # Must be bitwise or exactly equal
    assert torch.allclose(out1, out2, atol=1e-7)

    # Run 3 with different seed
    torch.manual_seed(9999)
    model3 = AttentionLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    model3.eval()
    with torch.no_grad():
        out3 = model3(x)

    assert not torch.allclose(out1, out3, atol=1e-3)


@pytest.mark.data
def test_lstm_data_causal_windows_on_cmapss() -> None:
    """Test window creation on actual C-MAPSS FD001 data."""
    from shikra.data.cmapss import load

    df_train = load("FD001", "train")
    sensors = get_informative_sensors()

    # Take first 2 units
    df_sub = df_train[df_train["unit"].isin([1, 2])].copy()
    scaler = StandardScaler()
    scaler.fit(df_sub[sensors].values)

    X, y, meta = create_causal_windows(df_sub, scaler, window_size=30, is_test=False, rul_cap=125)

    assert X.shape[1] == 30
    assert X.shape[2] == len(sensors)
    assert np.all(y <= 125.0)
    assert np.all(y >= 0.0)

    # Test set windowing (is_test=True)
    df_test = load("FD001", "test")
    df_test_sub = df_test[df_test["unit"].isin([1, 2])].copy()
    X_test, y_test, meta_test = create_causal_windows(df_test_sub, scaler, window_size=30, is_test=True)

    assert len(X_test) == 2
    assert X_test.shape == (2, 30, len(sensors))
    assert len(meta_test) == 2
