"""Performance and risk metrics for a market-making session."""
from __future__ import annotations

import numpy as np


def max_drawdown(equity):
    """Largest peak-to-trough drop of an equity curve (in price units)."""
    equity = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(equity)
    return float(np.max(running_max - equity))


def summarize(result):
    """Build a dict of headline metrics from a session result dict."""
    equity = result["equity"]
    fills = result["n_fills"]
    quotes = result["n_quotes"]
    inv = np.asarray(result["inventory_path"], dtype=float)
    return {
        "final_pnl": float(equity[-1]),
        "max_drawdown": max_drawdown(equity),
        "fill_rate": fills / quotes if quotes else 0.0,
        "n_fills": int(fills),
        "inventory_std": float(np.std(inv)),
        "max_abs_inventory": float(np.max(np.abs(inv))),
    }
