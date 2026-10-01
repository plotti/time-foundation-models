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
- `scripts/build_notebook.py` → `notebooks/tsfm_vs_classical.ipynb` ✅
  (run it on a **Colab T4 GPU**; the harness, rolling backtest, and metrics are verified).

The notebook runs the full ladder through one rolling-origin backtest, each model **with and
without covariates**, scored on point (MAE/MASE) **and** probabilistic (pinball/coverage)
metrics, and prints a copy-paste results block + blog draft.

> Note: the foundation-model cells and `statsforecast` need the fresh installs Colab
> provides — a local run can hit a `statsmodels`/`statsforecast` version clash. The backtest
> harness and metrics themselves are validated locally.

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
