# -*- coding: utf-8 -*-
"""Moirai (Salesforce) — isolated benchmark, run on its own.

Moirai's `uni2ts` package pins **torch 2.4.1**, which conflicts with the newer torch
that Chronos-2 and TimesFM 2.5 need. Running it in the same environment silently breaks
those models, so Moirai gets its own notebook. It uses the *identical* dataset, rolling
backtest, horizon, windows and metrics as the main benchmark — just paste its
RESULTS_JSON into the main leaderboard.
"""
# %% [markdown]
# # 🌙 Moirai (isolated) — Zurich Bike Counts
#
# Run this **separately** from the main `tsfm_vs_classical` notebook. `uni2ts` downgrades
# torch to 2.4.1 and would break Chronos-2 / TimesFM if installed alongside them.
#
# Same dataset, same 14-window rolling backtest, same metrics — so the numbers drop
# straight into the main table.
#
# > ⚡ Runtime → GPU (T4). Fresh runtime recommended (so the torch downgrade is clean).

# %% [markdown]
# ## 1 · Setup — uni2ts only
# %%
# !uv pip install -q --system uni2ts
import os, numpy as np, pandas as pd, torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)   # expect torch 2.4.x

# %% [markdown]
# ## 2 · Data + backtest harness (identical to the main notebook)
# %%
DATA_URL = "https://raw.githubusercontent.com/plotti/time-foundation-models/main/data/zurich_bikes.parquet"
if not os.path.exists("zurich_bikes.parquet"):
    import urllib.request; urllib.request.urlretrieve(DATA_URL, "zurich_bikes.parquet")

df = pd.read_parquet("zurich_bikes.parquet").sort_index()
df.index.name = "ds"
TARGET = "velo_count"
WEATHER = ["temp_c", "rain_min", "humidity_pct", "radiation", "wind_ms", "pressure_hpa"]
CALENDAR = ["hour", "dow", "month", "is_weekend", "covid"]
COVARIATES = WEATHER + CALENDAR

H = 24
N_WINDOWS = 14
STEP = 24
QUANTILES = [0.1, 0.5, 0.9]
CONTEXT = 24 * 60
origins = [len(df) - H - i * STEP for i in range(N_WINDOWS)][::-1]

def seasonal_naive_scale(y_hist, m=168):
    d = np.abs(y_hist[m:] - y_hist[:-m])
    return d.mean() if len(d) else np.abs(np.diff(y_hist)).mean()

def pinball(y, q_preds, quantiles):
    total = 0.0
    for qi, q in enumerate(quantiles):
        e = y - q_preds[:, qi]
        total += np.mean(np.maximum(q * e, (q - 1) * e))
    return total / len(quantiles)

def evaluate(name, forecaster, use_cov):
    import time
    t0 = time.time()
    maes, rmses, mases, pins, covs = [], [], [], [], []
    for o in origins:
        hist = df.iloc[max(0, o - CONTEXT):o]
        fut = df.iloc[o:o + H]
        y = fut[TARGET].values
        qp = forecaster(hist, fut[COVARIATES] if use_cov else None, H, QUANTILES)
        p50 = qp[:, QUANTILES.index(0.5)]
        maes.append(np.mean(np.abs(y - p50)))
        rmses.append(np.sqrt(np.mean((y - p50) ** 2)))
        mases.append(np.mean(np.abs(y - p50)) / seasonal_naive_scale(hist[TARGET].values))
        pins.append(pinball(y, qp, QUANTILES))
        covs.append(np.mean((y >= qp[:, 0]) & (y <= qp[:, -1])))
    tag = "＋cov" if use_cov else "base"
    r = dict(model=f"{name} {tag}", MAE=round(np.mean(maes)), RMSE=round(np.mean(rmses)),
             MASE=round(np.mean(mases), 3), pinball=round(np.mean(pins), 1),
             coverage=round(np.mean(covs), 2), secs=round(time.time() - t0, 1))
    print(f"  {r['model']:16} MAE {r['MAE']:>5} · MASE {r['MASE']:.3f} · "
          f"pinball {r['pinball']:>6} · cov {r['coverage']:.2f} ({r['secs']}s)")
    return r

RESULTS = []

# %% [markdown]
# ## 3 · Moirai — native any-variate covariates
# Real uni2ts/GluonTS API: wide PandasDataset (target NaN over the horizon) with
# `feat_dynamic_real` covariates, split off the last H, build MoiraiForecast with
# `feat_dynamic_real_dim`, then `create_predictor().predict()` and read `.quantile(q)`.
# %%
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

_moirai_mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

def moirai_fp_factory(with_x):
    def fp(hist, fut_cov, H, quantiles):
        feat_cols = COVARIATES if (with_x and fut_cov is not None) else []
        idx = hist.index.append(fut_cov.index if fut_cov is not None
                                else pd.date_range(hist.index[-1], periods=H + 1, freq="h")[1:])
        wide = pd.DataFrame(index=idx)
        wide[TARGET] = np.concatenate([hist[TARGET].values, np.full(H, np.nan)])
        for c in feat_cols:
            wide[c] = np.concatenate([hist[c].values, fut_cov[c].values])
        ds = PandasDataset(wide, target=TARGET, feat_dynamic_real=feat_cols or None, freq="h")
        _, test_tmpl = split(ds, offset=-H)
        test_data = test_tmpl.generate_instances(prediction_length=H)
        model = MoiraiForecast(
            module=_moirai_mod, prediction_length=H, context_length=len(hist),
            patch_size="auto", num_samples=100, target_dim=1,
            feat_dynamic_real_dim=len(feat_cols), past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=1)
        fc = next(iter(predictor.predict(test_data.input)))
        qs = np.column_stack([fc.quantile(q) for q in quantiles])
        return np.clip(qs, 0, None)
    return fp

RESULTS.append(evaluate("Moirai", moirai_fp_factory(False), False))
RESULTS.append(evaluate("Moirai", moirai_fp_factory(True), True))

# %% [markdown]
# ## 4 · 📋 Copy-paste results (fold into the main leaderboard)
# %%
import json
print("RESULTS_JSON_START")
print(json.dumps({"h": H, "windows": N_WINDOWS, "results": RESULTS}, indent=2))
print("RESULTS_JSON_END")
