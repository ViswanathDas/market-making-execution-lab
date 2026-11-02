"""Pins the README results (500 seeded paths) and checks the ablation logic."""
import pytest
import run_market_making as rmm
from src.market_maker import (SymmetricMarketMaker, AvellanedaStoikovMarketMaker,
                              SpreadMatchedSymmetricMaker)


@pytest.fixture(scope="module")
def runs():
    cfg = rmm.load_config()
    sym = rmm.run_batch(lambda c: SymmetricMarketMaker(c["maker"]["half_spread"]), cfg)
    sm = rmm.run_batch(lambda c: SpreadMatchedSymmetricMaker(
        gamma=c["maker"]["gamma"], sigma=c["market"]["sigma"], k=c["market"]["k"]), cfg)
    avs = rmm.run_batch(lambda c: AvellanedaStoikovMarketMaker(
        gamma=c["maker"]["gamma"], sigma=c["market"]["sigma"], k=c["market"]["k"]), cfg)
    return sym, sm, avs


def test_readme_numbers(runs):
    sym, _, avs = runs
    assert sym.final_pnl.std() == pytest.approx(13.0, abs=0.1)
    assert avs.final_pnl.std() == pytest.approx(6.5, abs=0.1)
    assert sym.max_drawdown.mean() == pytest.approx(6.1, abs=0.1)
    assert avs.max_drawdown.mean() == pytest.approx(1.2, abs=0.1)


def test_inventory_aware_roughly_halves_pnl_volatility(runs):
    sym, _, avs = runs
    assert avs.final_pnl.std() / sym.final_pnl.std() == pytest.approx(0.5, abs=0.05)
    assert sym.max_drawdown.mean() / avs.max_drawdown.mean() > 4.5


def test_skew_not_spread_drives_the_risk_reduction(runs):
    """Same spread without skew keeps almost all of the risk."""
    sym, sm, avs = runs
    assert sm.final_pnl.std() > 0.9 * sym.final_pnl.std()
    assert avs.final_pnl.std() < 0.6 * sm.final_pnl.std()


def test_spread_matched_quotes_are_centred():
    m = SpreadMatchedSymmetricMaker(gamma=0.1, sigma=2.0, k=1.5)
    bid, ask = m.quotes(100.0, inventory=10, time_left=0.5)
    assert (bid + ask) / 2 == pytest.approx(100.0)


def test_avellaneda_skews_against_inventory():
    m = AvellanedaStoikovMarketMaker(gamma=0.1, sigma=2.0, k=1.5)
    bid_long, ask_long = m.quotes(100.0, inventory=10, time_left=0.5)
    assert (bid_long + ask_long) / 2 < 100.0
