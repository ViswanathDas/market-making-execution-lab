"""Execution algorithms and implementation shortfall.

A parent order to buy ``total_qty`` shares is sliced into child orders.
``twap`` spreads the order evenly across the horizon. Each child pays the
mid-price plus a temporary impact proportional to its participation rate,
so trading faster costs more. Implementation shortfall is measured against
the arrival (decision) mid-price.
"""
from __future__ import annotations

import numpy as np

from .market_sim import MarketSimulator


def twap(total_qty, n_slices):
    """Even TWAP schedule: equal child size in each slice."""
    base = total_qty // n_slices
    sched = np.full(n_slices, base, dtype=float)
    sched[: total_qty - base * n_slices] += 1  # spread the remainder
    return sched


def execute(schedule, mid, impact_coeff=0.02, adv_per_slice=200.0):
    """Execute a buy schedule and return fills and average price.

    Temporary impact per child = impact_coeff * (child_qty / adv_per_slice),
    added to the mid-price at that slice.
    """
    n = len(schedule)
    idx = np.linspace(0, len(mid) - 1, n).astype(int)
    fill_prices = np.empty(n)
    for i, q in enumerate(schedule):
        impact = impact_coeff * (q / adv_per_slice)
        fill_prices[i] = mid[idx[i]] + impact
    filled = float(np.sum(schedule))
    avg_price = float(np.sum(fill_prices * schedule) / filled)
    return {"avg_price": avg_price, "fill_prices": fill_prices,
            "schedule": schedule, "slice_idx": idx}


def implementation_shortfall(avg_price, arrival_price, total_qty):
    """Cost vs. trading the whole order at the arrival price (price units)."""
    return (avg_price - arrival_price) * total_qty
