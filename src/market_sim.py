"""Stochastic mid-price and order-flow simulation.

The mid-price follows an arithmetic Brownian motion (constant volatility,
no drift by default), which is the standard reference model in the
market-making literature (Avellaneda & Stoikov, 2008). Market orders
arrive as a Poisson process whose intensity decays exponentially with the
distance of a quote from the mid-price, so tighter quotes fill more often.
"""
from __future__ import annotations

import numpy as np


class MarketSimulator:
    """Generate a mid-price path and the order-flow fill model.

    Parameters
    ----------
    s0 : float
        Initial mid-price.
    sigma : float
        Volatility of the mid-price per unit time (price units).
    dt : float
        Time step.
    n_steps : int
        Number of steps in the session.
    A : float
        Base arrival intensity of market orders at distance 0.
    k : float
        Decay of fill probability with quote distance (larger k = pickier
        counterparties, fewer fills for the same edge).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, s0=100.0, sigma=2.0, dt=1.0 / 200, n_steps=200,
                 A=140.0, k=1.5, seed=0):
        self.s0 = s0
        self.sigma = sigma
        self.dt = dt
        self.n_steps = n_steps
        self.A = A
        self.k = k
        self.rng = np.random.default_rng(seed)

    def mid_price_path(self):
        """Return an array of mid-prices of length n_steps + 1."""
        shocks = self.rng.normal(0.0, self.sigma * np.sqrt(self.dt), self.n_steps)
        path = np.empty(self.n_steps + 1)
        path[0] = self.s0
        path[1:] = self.s0 + np.cumsum(shocks)
        return path

    def fill_probability(self, distance):
        """Probability a resting quote at `distance` from mid is hit in dt.

        Uses the exponential intensity lambda = A * exp(-k * distance) and
        converts to a per-step probability 1 - exp(-lambda * dt).
        """
        distance = np.maximum(distance, 0.0)
        intensity = self.A * np.exp(-self.k * distance)
        return 1.0 - np.exp(-intensity * self.dt)

    def draw_fill(self, distance):
        """Bernoulli draw of whether a quote at `distance` is filled."""
        return self.rng.random() < self.fill_probability(distance)
