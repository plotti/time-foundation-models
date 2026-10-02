# Time Series Foundation Models — can they beat ARIMA & Prophet when covariates matter?

The **time-series sibling** of the
[tabular-models-vs-xgboost](https://github.com/plotti/tabular-models-vs-xgboost) benchmark.

Same recipe, new domain: take a real dataset where **multiple inputs drive a target
series**, and run a fair ladder of models at it —
**classical → intermediate → foundation** — under one identical backtesting protocol.

## The ladder

| Tier | Models | Uses covariates? |
|---|---|---|
| **Classical** | Seasonal Naive, **SARIMAX** (ARIMA + exogenous), ETS | SARIMAX ✅ |
| **Intermediate** | **Prophet** (+ regressors), LightGBM on lag features | ✅ |
| **Foundation** | **Chronos-2**, **TabPFN-TS**, TimesFM 2.5, Moirai | Chronos-2 & TabPFN-TS ✅ natively |

The real question isn't just "foundation model vs. ARIMA" — it's **how much do the
exogenous inputs actually help each tier?** (The time-series echo of the energy-class
feature from the tabular project.)

## Status

🛠️ **Dataset + notebook built.** Full research, dataset analysis, baseline rationale, and
evaluation design in **[`research/RESEARCH.md`](research/RESEARCH.md)**.

- `scripts/build_dataset.py` → `data/zurich_bikes.parquet` (43,762 hourly rows, 2019–2023) ✅
- `scripts/tsfm_vs_classical.py` → **source of truth** (runnable, jupytext `# %%` cells) ✅
- `scripts/py_to_notebook.py` → `notebooks/tsfm_vs_classical.ipynb` (generated) ✅

Run the notebook on a **Colab T4 GPU**. It runs the full ladder through one rolling-origin
backtest, each model **with and without covariates**, scored on point (MAE/MASE) **and**
probabilistic (pinball/coverage) metrics, then prints a copy-paste results block + blog draft.

### Verified locally (ran on the real data, clean venvs)

| Model | base MASE | +cov MASE | status |
|---|---|---|---|
| SeasonalNaive | 0.349 | — | ✅ ran |
| AutoETS | 0.914 | — | ✅ ran |
| ARIMA/SARIMAX | 0.536 | 0.582 | ✅ ran |
| LightGBM | 0.541 | **0.333** | ✅ ran (cov −38%) |
| TimesFM 2.5 | ~0.41 | — | ✅ ran |
| **TabPFN-TS** | 0.426 | **0.250** | ✅ ran (cov −41%) |
| **Moirai** | 0.489 | 0.614 | ✅ ran (cov hurt) |
| Chronos-2 | — | — | GPU-only, run on Colab |

(Numbers are a few windows for verification, not the full 14-window benchmark.)

**Recommended dataset:** EPF (Electricity Price Forecasting) — hourly day-ahead prices for
5 European/US markets with *future-known* load & wind covariates. Clean target/covariate
split, genuinely covariate-driven, mature download path.

## Layout

```
research/   RESEARCH.md — dataset candidates, baseline ladder, eval protocol, sources
data/       downloaded + assembled datasets (gitignored)
scripts/    dataset builders, benchmark runners
notebooks/  the annotated Colab benchmark (to come)
```

## Next steps

See the open questions at the end of `research/RESEARCH.md` — dataset choice, which
foundation models to include, and point-vs-probabilistic evaluation — then build
`scripts/build_dataset.py` and the notebook.
