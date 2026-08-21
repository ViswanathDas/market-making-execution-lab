"""TWAP execution and implementation shortfall.

Executes the same parent buy order at several speeds (number of slices).
Because a single price path is dominated by random drift, the shortfall is
averaged over many seeded paths so the drift washes out and the residual is
the temporary market impact: trading faster (fewer, larger child orders)
pays more impact. Shortfall is measured against the arrival mid-price.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.market_sim import MarketSimulator
from src.execution import twap, execute, implementation_shortfall

HERE = os.path.dirname(__file__)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    total_qty = 2000
    n_paths = 400
    speeds = [1, 5, 10, 20, 40, 80]

    rows = []
    for n_slices in speeds:
        sched = twap(total_qty, n_slices)
        shortfalls_bps = []
        for i in range(n_paths):
            sim = MarketSimulator(s0=100.0, sigma=2.0, dt=0.005, n_steps=200, seed=1000 + i)
            mid = sim.mid_price_path()
            arrival = mid[0]
            ex = execute(sched, mid, impact_coeff=0.25, adv_per_slice=500.0)
            shortfalls_bps.append(1e4 * (ex["avg_price"] - arrival) / arrival)
        rows.append({"n_slices": n_slices,
                     "child_qty": total_qty / n_slices,
                     "mean_shortfall_bps": float(np.mean(shortfalls_bps)),
                     "std_shortfall_bps": float(np.std(shortfalls_bps))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "execution.csv"), index=False)
    print(df.to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.errorbar(df.n_slices, df.mean_shortfall_bps, yerr=df.std_shortfall_bps,
                 marker="o", capsize=3)
    plt.xlabel("Number of TWAP slices (more = slower, smaller children)")
    plt.ylabel("Mean implementation shortfall (bps)")
    plt.title("Execution speed vs impact cost (avg over %d paths)" % n_paths)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "results", "execution_shortfall.png"), dpi=110)
    plt.close()
    print("\nSaved results/execution.csv, results/execution_shortfall.png")


if __name__ == "__main__":
    main()
