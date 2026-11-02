"""Compare an inventory-aware maker against a symmetric baseline.

Runs both strategies over many seeded price paths and reports the PnL
distribution, drawdown, fill rate, and inventory risk. Saves a summary
CSV and two figures to results/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.market_sim import MarketSimulator
from src.market_maker import (SymmetricMarketMaker, AvellanedaStoikovMarketMaker,
                              SpreadMatchedSymmetricMaker)
from src.backtest import run_session
from src.metrics import summarize

HERE = os.path.dirname(__file__)


def load_config():
    with open(os.path.join(HERE, "configs", "base.yaml")) as f:
        return yaml.safe_load(f)


def run_batch(maker_factory, cfg):
    m, r = cfg["market"], cfg["risk"]
    rows = []
    for i in range(cfg["experiment"]["n_paths"]):
        seed = cfg["experiment"]["base_seed"] + i
        sim = MarketSimulator(s0=m["s0"], sigma=m["sigma"], dt=m["dt"],
                              n_steps=m["n_steps"], A=m["A"], k=m["k"], seed=seed)
        res = run_session(maker_factory(cfg), sim, max_inventory=r["max_inventory"])
        rows.append(summarize(res))
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    sym = run_batch(lambda c: SymmetricMarketMaker(c["maker"]["half_spread"]), cfg)
    avs = run_batch(lambda c: AvellanedaStoikovMarketMaker(
        gamma=c["maker"]["gamma"], sigma=c["market"]["sigma"], k=c["market"]["k"]), cfg)

    # Ablation: same spread as Avellaneda-Stoikov, no inventory skew.
    sm = run_batch(lambda c: SpreadMatchedSymmetricMaker(
        gamma=c["maker"]["gamma"], sigma=c["market"]["sigma"], k=c["market"]["k"]), cfg)

    runs = {"symmetric": sym, "spread_matched_no_skew": sm, "avellaneda_stoikov": avs}
    summary = pd.DataFrame({
        "strategy": list(runs),
        "mean_pnl": [d.final_pnl.mean() for d in runs.values()],
        "std_pnl": [d.final_pnl.std() for d in runs.values()],
        "sharpe_like": [d.final_pnl.mean() / d.final_pnl.std() for d in runs.values()],
        "mean_drawdown": [d.max_drawdown.mean() for d in runs.values()],
        "mean_fill_rate": [d.fill_rate.mean() for d in runs.values()],
        "mean_inventory_std": [d.inventory_std.mean() for d in runs.values()],
        "mean_max_abs_inv": [d.max_abs_inventory.mean() for d in runs.values()],
    })
    summary.to_csv(os.path.join(HERE, "results", "summary.csv"), index=False)
    print(summary.to_string(index=False))

    # Figure 1: PnL distributions
    plt.figure(figsize=(7, 4))
    bins = np.linspace(min(sym.final_pnl.min(), avs.final_pnl.min()),
                       max(sym.final_pnl.max(), avs.final_pnl.max()), 40)
    plt.hist(sym.final_pnl, bins=bins, alpha=0.55, label="symmetric")
    plt.hist(avs.final_pnl, bins=bins, alpha=0.55, label="inventory-aware")
    plt.xlabel("Final PnL (price units)")
    plt.ylabel("Frequency")
    plt.title("PnL distribution over %d paths" % cfg["experiment"]["n_paths"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "results", "pnl_distribution.png"), dpi=110)
    plt.close()

    # Figure 2: inventory risk vs PnL std
    plt.figure(figsize=(7, 4))
    plt.scatter(sym.inventory_std, sym.final_pnl, s=8, alpha=0.4, label="symmetric")
    plt.scatter(avs.inventory_std, avs.final_pnl, s=8, alpha=0.4, label="inventory-aware")
    plt.xlabel("Inventory std over session")
    plt.ylabel("Final PnL")
    plt.title("Inventory risk vs PnL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "results", "inventory_vs_pnl.png"), dpi=110)
    plt.close()
    print("\nSaved results/summary.csv, results/pnl_distribution.png, results/inventory_vs_pnl.png")


if __name__ == "__main__":
    main()
