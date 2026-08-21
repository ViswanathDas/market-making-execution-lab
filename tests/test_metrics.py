"""Basic sanity tests. Run with: python -m pytest -q"""
import numpy as np
from src.metrics import max_drawdown
from src.execution import twap, execute, implementation_shortfall
from src.market_sim import MarketSimulator


def test_max_drawdown_simple():
    assert max_drawdown([1, 2, 3]) == 0.0
    assert max_drawdown([3, 1, 4, 2]) == 2.0


def test_twap_conserves_quantity():
    assert twap(2000, 7).sum() == 2000
    assert twap(2001, 8).sum() == 2001


def test_fill_probability_monotonic():
    sim = MarketSimulator(seed=1)
    assert sim.fill_probability(0.0) > sim.fill_probability(1.0) > sim.fill_probability(2.0)


def test_shortfall_zero_when_no_impact_and_flat_price():
    mid = np.full(50, 100.0)
    sched = twap(1000, 10)
    ex = execute(sched, mid, impact_coeff=0.0)
    assert abs(implementation_shortfall(ex["avg_price"], 100.0, 1000)) < 1e-9
