"""Event loop that runs a quoting policy over a simulated session."""
from __future__ import annotations

import numpy as np

from .market_sim import MarketSimulator


def run_session(maker, sim: MarketSimulator, max_inventory=20):
    """Run one market-making session.

    At each step the maker posts a bid and an ask. Each side may be filled
    independently according to the simulator's distance-based fill model.
    Cash and inventory are updated on fills; the maker is prevented from
    quoting a side that would breach the inventory limit. PnL is marked to
    the mid-price (cash + inventory * mid).
    """
    mid = sim.mid_price_path()
    dt = sim.dt
    n = sim.n_steps

    cash = 0.0
    inventory = 0
    equity = np.empty(n + 1)
    inv_path = np.empty(n + 1)
    equity[0] = 0.0
    inv_path[0] = 0
    n_quotes = 0
    n_fills = 0

    for t in range(n):
        time_left = (n - t) * dt
        bid, ask = maker.quotes(mid[t], inventory, time_left)

        # Buy side (our bid). Skip if it would breach the long limit.
        if inventory < max_inventory:
            n_quotes += 1
            if sim.draw_fill(mid[t] - bid):
                cash -= bid
                inventory += 1
                n_fills += 1
        # Sell side (our ask). Skip if it would breach the short limit.
        if inventory > -max_inventory:
            n_quotes += 1
            if sim.draw_fill(ask - mid[t]):
                cash += ask
                inventory -= 1
                n_fills += 1

        equity[t + 1] = cash + inventory * mid[t + 1]
        inv_path[t + 1] = inventory

    return {
        "mid": mid,
        "equity": equity,
        "inventory_path": inv_path,
        "n_quotes": n_quotes,
        "n_fills": n_fills,
        "final_cash": cash,
        "final_inventory": inventory,
    }
