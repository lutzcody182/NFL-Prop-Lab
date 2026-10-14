"""Odds conversions, margin removal, expected value and closing-line value.

All probabilities are in [0, 1]. American odds: +150 means win $150 on $100;
-120 means stake $120 to win $100. Decimal odds include the stake (2.50 = +150).
"""
from __future__ import annotations

import math


def american_to_decimal(a: float) -> float:
    if a == 0 or -100 < a < 100:
        raise ValueError(f"not a valid American price: {a}")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(d: float) -> float:
    if d <= 1:
        raise ValueError(f"decimal odds must be > 1: {d}")
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def implied(d: float) -> float:
    """Implied probability of a decimal price (includes the book's margin)."""
    return 1 / d


def overround(*decimals: float) -> float:
    """Sum of implied probabilities minus 1, e.g. 0.045 for a -110/-110 market."""
    return sum(1 / d for d in decimals) - 1


# ---------- margin removal for a two-way market ----------

def novig_multiplicative(d_over: float, d_under: float) -> float:
    """Scale both implied probabilities so they sum to 1. Returns fair P(over)."""
    po, pu = 1 / d_over, 1 / d_under
    return po / (po + pu)


def novig_power(d_over: float, d_under: float, tol: float = 1e-12) -> float:
    """Find k with p_over^k + p_under^k = 1 (shades longshots more). Returns fair P(over)."""
    po, pu = 1 / d_over, 1 / d_under
    lo, hi = 1.0, 10.0
    if po + pu <= 1:  # no margin (or negative): fall back to normalizing
        return po / (po + pu)
    for _ in range(200):
        k = (lo + hi) / 2
        s = po ** k + pu ** k
        if abs(s - 1) < tol:
            break
        if s > 1:
            lo = k
        else:
            hi = k
    return po ** k


def novig_shin(d_over: float, d_under: float, tol: float = 1e-12) -> float:
    """Shin (1993) method for a two-outcome market. Returns fair P(over)."""
    pi = [1 / d_over, 1 / d_under]
    total = sum(pi)
    if total <= 1:
        return pi[0] / total

    def fair(z: float) -> list[float]:
        return [(math.sqrt(z * z + 4 * (1 - z) * p * p / total) - z) / (2 * (1 - z)) for p in pi]

    lo, hi = 0.0, 0.5
    for _ in range(200):
        z = (lo + hi) / 2
        s = sum(fair(z))
        if abs(s - 1) < tol:
            break
        if s > 1:
            lo = z
        else:
            hi = z
    return fair(z)[0]


NOVIG = {"multiplicative": novig_multiplicative, "power": novig_power, "shin": novig_shin}


# ---------- value ----------

def expected_value(q: float, d: float) -> float:
    """Expected profit per 1 unit staked at decimal odds d when the true win probability is q."""
    return q * (d - 1) - (1 - q)


def fair_decimal(q: float) -> float:
    return 1 / q


def clv_prob(p_bet: float, p_close: float, side: str) -> float:
    """Closing-line value in probability points for the side taken at bet time:
    positive when the closing fair probability of that side is higher than at bet time."""
    if side == "over":
        return p_close - p_bet
    if side == "under":
        return p_bet - p_close
    raise ValueError(side)


def log_loss(p: float, outcome: int, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return -math.log(p) if outcome else -math.log(1 - p)
