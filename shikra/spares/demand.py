"""Demand pre-positioning utility for SHIKRA fleet simulation.

Provides the ``removal_prob`` function used by the shikra and oracle policies
to decide whether to pre-position a spare engine from depot to base.

Public API
----------
removal_prob(q10, q50, q90, h) -> float
"""

from __future__ import annotations

import numpy as np


def removal_prob(q10: float, q50: float, q90: float, h: float) -> float:
    """Estimate the probability that an engine needs removal within horizon h cycles.

    Uses a piecewise-linear CDF fitted through:
        (q10 → 0.10,  q50 → 0.50,  q90 → 0.90)

    Then returns P(RUL ≤ h) = CDF(h), clipped to [0, 1].

    Parameters
    ----------
    q10, q50, q90 : RUL quantile predictions for the current cycle
    h             : planning horizon in cycles

    Returns
    -------
    float in [0, 1] — estimated probability of needing removal within h cycles
    """
    if q90 <= q10:
        # Degenerate (oracle or near-oracle): point mass at the quantile value
        return 1.0 if h >= q50 else 0.0

    # Piecewise-linear interpolation through three (rul, cdf) knot points
    knots = [(float(q10), 0.10), (float(q50), 0.50), (float(q90), 0.90)]

    h = float(h)

    if h <= knots[0][0]:
        # Below q10 knot — extrapolate from (0, 0) to (q10, 0.10)
        if knots[0][0] <= 0:
            return 0.10
        return float(np.clip(0.10 * h / knots[0][0], 0.0, 0.10))

    if h >= knots[2][0]:
        # Above q90 knot — extrapolate from (q90, 0.90) to (∞, 1.0)
        # We cap at the segment slope and clip
        excess = h - knots[2][0]
        span = knots[2][0] - knots[1][0]
        if span <= 0:
            return 1.0
        slope = (1.0 - 0.90) / span   # approximate slope beyond q90
        return float(np.clip(0.90 + slope * excess, 0.0, 1.0))

    # Interpolate between the three knots
    for i in range(len(knots) - 1):
        x0, y0 = knots[i]
        x1, y1 = knots[i + 1]
        if x0 <= h <= x1:
            if x1 == x0:
                return float(y1)
            frac = (h - x0) / (x1 - x0)
            return float(np.clip(y0 + frac * (y1 - y0), 0.0, 1.0))

    return 1.0  # fallback


def inverse_removal_prob(q10: float, q50: float, q90: float, u: float) -> float:
    """Sample remaining life from the inverse piecewise-linear CDF."""
    u = float(np.clip(u, 0.0, 1.0))
    q10, q50, q90 = float(q10), float(q50), float(q90)

    if q90 <= q10:
        return max(0.0, q50)
    if u <= 0.10:
        return max(0.0, (u / 0.10) * q10) if q10 > 0 else 0.0
    if u <= 0.50:
        return max(0.0, q10 + ((u - 0.10) / 0.40) * (q50 - q10))
    if u <= 0.90:
        return max(0.0, q50 + ((u - 0.50) / 0.40) * (q90 - q50))

    span = q90 - q50
    if span <= 0:
        return max(0.0, q90)
    return max(0.0, q90 + (u - 0.90) * span / 0.10)
