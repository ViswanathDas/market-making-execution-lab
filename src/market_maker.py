"""Inventory-aware market-making strategies.

Two quoting policies are implemented:

* ``SymmetricMarketMaker`` posts a fixed half-spread around the mid-price and
  ignores inventory. It is the naive baseline.

* ``AvellanedaStoikovMarketMaker`` skews its quotes using a reservation price
  that shifts away from the current inventory, and sets the spread from the
  risk-aversion and the order-flow decay. This is the classic inventory-aware
  policy and is the strategy the lab is built to study.
"""
from __future__ import annotations

import numpy as np


class SymmetricMarketMaker:
    """Fixed half-spread around the mid, no inventory control."""

    def __init__(self, half_spread=0.6):
        self.half_spread = half_spread

    def quotes(self, mid, inventory, time_left):
        return mid - self.half_spread, mid + self.half_spread


class AvellanedaStoikovMarketMaker:
    """Inventory-aware quoting (Avellaneda & Stoikov, 2008).

    Parameters
    ----------
    gamma : float
        Risk aversion. Higher gamma skews quotes harder against inventory.
    sigma : float
        Mid-price volatility (same units as the simulator).
    k : float
        Order-flow decay used by the simulator's fill model.
    """

    def __init__(self, gamma=0.1, sigma=2.0, k=1.5):
        self.gamma = gamma
        self.sigma = sigma
        self.k = k

    def reservation_price(self, mid, inventory, time_left):
        """Inventory-adjusted fair value the maker actually quotes around."""
        return mid - inventory * self.gamma * self.sigma ** 2 * time_left

    def optimal_spread(self, time_left):
        """Total bid-ask spread from risk aversion and order-flow decay."""
        return (self.gamma * self.sigma ** 2 * time_left
                + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k))

    def quotes(self, mid, inventory, time_left):
        r = self.reservation_price(mid, inventory, time_left)
        half = 0.5 * self.optimal_spread(time_left)
        return r - half, r + half
