# Market Making and Execution Lab

[![tests](https://github.com/ViswanathDas/market-making-execution-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/ViswanathDas/market-making-execution-lab/actions/workflows/tests.yml)

A small, self-contained Python lab for studying two questions a trading desk
cares about:

1. **How should a market maker quote when it has to hold inventory?**
2. **How fast should a large order be executed given market impact?**

Everything runs from simulated data with seeded random paths, so results are
reproducible and the moving parts are visible.

## What is inside

```
market-making-execution-lab/
├── src/
│   ├── market_sim.py     # mid-price (arithmetic Brownian motion) + order-flow fill model
│   ├── market_maker.py   # symmetric baseline + Avellaneda-Stoikov inventory-aware quoting
│   ├── backtest.py       # event loop: quote, fill, update cash and inventory, mark to mid
│   ├── execution.py      # TWAP schedule, temporary market impact, implementation shortfall
│   └── metrics.py        # PnL, max drawdown, fill rate, inventory risk
├── configs/base.yaml     # all parameters in one place
├── run_market_making.py  # compares the two makers over many paths, saves figures + CSV
├── run_execution.py      # execution speed vs impact cost, saves figure + CSV
├── tests/               # sanity tests + tests that pin the README results
└── results/              # generated CSVs and PNGs
```

## The models

**Order flow.** Market orders arrive with intensity `lambda = A * exp(-k * d)`,
where `d` is the distance of a quote from the mid-price. Tighter quotes fill
more often. This is the fill model from Avellaneda and Stoikov (2008).

**Inventory-aware quoting.** Instead of quoting symmetrically around the mid,
the maker quotes around a *reservation price* that shifts away from its current
inventory, `r = mid - q * gamma * sigma^2 * (T - t)`, and sets its spread from
risk aversion `gamma` and the order-flow decay `k`. When the maker is long it
lowers its quotes to sell down; when short it raises them to buy back.

**Execution.** A parent order is sliced by TWAP. Each child pays the mid-price
plus a temporary impact proportional to its size relative to available depth,
so trading faster costs more. Implementation shortfall is measured against the
arrival (decision) price.

## Results (500 paths, seed 0)

| strategy | mean PnL | std PnL | mean drawdown | fill rate | inventory std |
|---|---:|---:|---:|---:|---:|
| symmetric (half spread 0.6) | 59.0 | 13.0 | 6.1 | 0.25 | 3.19 |
| spread matched, no skew (ablation) | 60.1 | 12.6 | 5.4 | 0.20 | 2.97 |
| inventory-aware (Avellaneda-Stoikov) | 56.8 | 6.5 | 1.2 | 0.21 | 1.36 |

The inventory-aware maker gives up a little mean PnL but roughly halves PnL
volatility and cuts mean drawdown about 5x by holding far less inventory. Its
Sharpe-like ratio (mean/std) is about 8.7 versus 4.6.

**Is it the skew or the wider spread?** The Avellaneda-Stoikov maker also quotes
wider than the baseline, so the comparison alone cannot say which part does the
work. The ablation keeps the Avellaneda-Stoikov spread but centres it on the mid.
It keeps almost all of the baseline's risk (std 12.6), so the risk reduction
comes from skewing quotes against inventory, not from the wider spread.

**Execution.** Implementation shortfall falls from 100 bps for a single large
order to about 3 bps when sliced into 80 children. The standard deviation of the
shortfall (about 110 bps once the order is sliced) is timing risk from the price
moving during execution, and it grows slightly as execution slows. That is the
impact-versus-timing-risk trade-off.

### Assumptions and limits

- Everything is simulated: arithmetic Brownian motion mid-price, Poisson order
  arrivals with intensity `A exp(-k d)`, no adverse selection (fills do not
  predict the next price move), no fees or latency, and a hard inventory cap of 20.
- Execution uses a linear temporary impact model and TWAP only; no permanent
  impact and no optimal (Almgren-Chriss) schedule yet.
- Results are a study of the mechanics, not a trading claim.

## Tests

`python -m pytest -q` runs the sanity tests and pins the numbers in the table
above (500 seeded paths), including the ablation result. GitHub Actions runs
them on every push.

## Run it

```bash
pip install -r requirements.txt
python run_market_making.py     # writes results/summary.csv + two figures
python run_execution.py         # writes results/execution.csv + one figure
python -m pytest -q             # sanity tests
```

## Reference

Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit order
book.* Quantitative Finance, 8(3), 217-224.
