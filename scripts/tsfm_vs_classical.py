# -*- coding: utf-8 -*-
"""Time Series Foundation Models vs. ARIMA & Prophet — Zurich Bike Counts.

Standalone, runnable version of the benchmark notebook. Source of truth; the
Jupyter notebook is generated from this via scripts/py_to_notebook.py.

Model ladder, each run with and without weather/calendar covariates, through one
rolling-origin backtest scored on point (MAE/MASE) and probabilistic (pinball/
coverage) metrics:
  classical   : SeasonalNaive, AutoARIMA/SARIMAX, AutoETS  (statsforecast)
  intermediate: Prophet, LightGBM                          (prophet, mlforecast)
  foundation  : Chronos-2, TimesFM 2.5, Moirai, TabPFN-TS

Run on a GPU (Colab T4). TabPFN-TS needs a free Prior Labs token.
"""
# %% [markdown]
# # 🚲 Time Series Foundation Models vs. ARIMA & Prophet — Zurich Bike Counts
#
# The time-series sibling of the
# [tabular-models-vs-xgboost](https://github.com/plotti/tabular-models-vs-xgboost) benchmark.
#
# **The question:** can zero-shot *time-series foundation models* (Chronos-2, TabPFN-TS,
# TimesFM, Moirai) beat classical **ARIMA**, **Prophet**, and a tuned **LightGBM** at
# forecasting hourly bicycle traffic in Zurich — and **how much do weather covariates
# (temperature, rain) actually help each one?**
#
# > ⚡ Runtime → Change runtime type → GPU (T4) before running.

# %% [markdown]
# ## 1 · Setup
# %%
# !uv pip install -q --system statsforecast mlforecast utilsforecast prophet lightgbm \
#     chronos-forecasting tabpfn-time-series tabpfn-client "timesfm[xreg]" "jax[cpu]"
# NOTE: Moirai (uni2ts) is intentionally NOT installed here — it pins torch 2.4.1 and
# breaks Chronos-2 / TimesFM. It lives in its own notebook (moirai_only.ipynb).
import os, numpy as np, pandas as pd
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)
if DEVICE == "cpu":
    print("⚠️  No GPU — Runtime → Change runtime type → T4 GPU, then rerun.")

# %% [markdown]
# ### TabPFN token (for TabPFN-TS only)
# Free token from [priorlabs.ai](https://priorlabs.ai) → Account → API Key.
# Leave blank to skip TabPFN-TS; every other model still runs.
# %%
TABPFN_TOKEN = ""  # <-- paste tabpfn_sk_... here
HAVE_TABPFN = bool(TABPFN_TOKEN.strip())
if HAVE_TABPFN:
    os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN.strip()
    try:
        import tabpfn_client
        tabpfn_client.set_access_token(TABPFN_TOKEN.strip())
    except Exception as e:
        print("tabpfn_client note:", e)
    print("✅ TabPFN token set")
else:
    print("ℹ️  No token — TabPFN-TS will be skipped.")

# %% [markdown]
# ## 2 · Get the data
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
print(f"{len(df):,} hourly rows · {df.index.min()} → {df.index.max()}")
print(f"target: {TARGET} (median {df[TARGET].median():.0f}/h, max {df[TARGET].max():.0f})")

# %% [markdown]
# ### The signal we're hoping the models find
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
wd = df[df.is_weekend == 0].groupby("hour")[TARGET].mean()
we = df[df.is_weekend == 1].groupby("hour")[TARGET].mean()
ax[0].plot(wd.index, wd.values, label="weekday", lw=2)
ax[0].plot(we.index, we.values, label="weekend", lw=2)
ax[0].set_title("Mean bikes/hour by hour"); ax[0].set_xlabel("hour"); ax[0].legend()
peak = df[(df.is_weekend == 0) & (df.hour.isin([7, 8, 17, 18]))]
dry = peak[peak.rain_min == 0][TARGET].mean(); wet = peak[peak.rain_min > 0][TARGET].mean()
ax[1].bar(["dry", "rainy"], [dry, wet], color=["#2a9d8f", "#6699cc"])
ax[1].set_title(f"Commute-hour cycling: rain drops it {100*(1-wet/dry):.0f}%")
for i, v in enumerate([dry, wet]): ax[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3 · Evaluation protocol — rolling-origin backtest
# %%
H = 24                 # forecast horizon (hours)
N_WINDOWS = 14         # rolling cutoffs
STEP = 24              # hours between cutoffs
QUANTILES = [0.1, 0.5, 0.9]
CONTEXT = 24 * 60      # history each model gets (~60 days)

origins = [len(df) - H - i * STEP for i in range(N_WINDOWS)][::-1]
print(f"H={H}h · {N_WINDOWS} windows · context={CONTEXT}h")
print(f"first origin {df.index[origins[0]]} … last {df.index[origins[-1]]}")

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
    """forecaster(hist_df, future_cov_df, H, quantiles) -> (H, n_quantiles) array."""
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
    print(f"  {r['model']:26} MAE {r['MAE']:>5} · MASE {r['MASE']:.3f} · "
          f"pinball {r['pinball']:>6} · cov {r['coverage']:.2f} ({r['secs']}s)")
    return r

RESULTS = []

# %% [markdown]
# ## 4 · Classical baselines — Seasonal-Naive, ARIMA, ETS
# %%
# !uv pip install -q --system statsforecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA, AutoETS

def sf_forecaster(model_ctor, with_x):
    def fp(hist, fut_cov, H, quantiles):
        use_x = with_x
        sdf = pd.DataFrame({"unique_id": "z", "ds": hist.index, "y": hist[TARGET].values})
        levels = [80]  # -> lo-80/hi-80 = P10/P90 band
        kw = {}
        if use_x and fut_cov is not None:
            # drop covariates that are constant in this window (SARIMAX chokes on them)
            active = [c for c in COVARIATES if hist[c].nunique() > 1]
            if active:
                for c in active:
                    sdf[c] = hist[c].values
                X_df = fut_cov.reset_index(); X_df["unique_id"] = "z"
                kw["X_df"] = X_df[["unique_id", "ds"] + active]
            else:
                use_x = False
        sf = StatsForecast(models=[model_ctor], freq="h", n_jobs=1)
        fc = sf.forecast(df=sdf, h=H, level=levels, **kw)
        name = model_ctor.__class__.__name__
        p50 = fc[name].values
        lo = fc[f"{name}-lo-80"].values; hi = fc[f"{name}-hi-80"].values
        return np.clip(np.column_stack([lo, p50, hi]), 0, None)
    return fp

# AutoARIMA search space capped so it doesn't take forever per window
ARIMA = lambda: AutoARIMA(season_length=24, max_p=5, max_d=2, max_q=5,
                          max_P=2, max_D=1, max_Q=2)
RESULTS.append(evaluate("SeasonalNaive", sf_forecaster(SeasonalNaive(season_length=168), False), False))
RESULTS.append(evaluate("AutoETS", sf_forecaster(AutoETS(season_length=24), False), False))
RESULTS.append(evaluate("ARIMA", sf_forecaster(ARIMA(), False), False))
RESULTS.append(evaluate("ARIMA", sf_forecaster(ARIMA(), True), True))

# %% [markdown]
# ## 5 · Intermediate — Prophet & LightGBM
# %%
# !uv pip install -q --system prophet
from prophet import Prophet

def prophet_fp(hist, fut_cov, H, quantiles):
    p = Prophet(interval_width=0.8, daily_seasonality=True, weekly_seasonality=True,
                yearly_seasonality=True)
    pdf = pd.DataFrame({"ds": hist.index, "y": hist[TARGET].values})
    if fut_cov is not None:
        for c in WEATHER:
            p.add_regressor(c); pdf[c] = hist[c].values
    p.fit(pdf)
    fdf = pd.DataFrame({"ds": fut_cov.index if fut_cov is not None
                        else pd.date_range(hist.index[-1], periods=H + 1, freq="h")[1:]})
    if fut_cov is not None:
        for c in WEATHER:
            fdf[c] = fut_cov[c].values
    fc = p.predict(fdf)
    return np.clip(np.column_stack([fc["yhat_lower"], fc["yhat"], fc["yhat_upper"]]), 0, None)

RESULTS.append(evaluate("Prophet", prophet_fp, False))
RESULTS.append(evaluate("Prophet", prophet_fp, True))

# %%
# !uv pip install -q --system mlforecast lightgbm
from mlforecast import MLForecast
import lightgbm as lgb

def lgb_fp_factory(with_x):
    def fp(hist, fut_cov, H, quantiles):
        sdf = pd.DataFrame({"unique_id": "z", "ds": hist.index, "y": hist[TARGET].values})
        xcols = []
        if with_x and fut_cov is not None:
            for c in COVARIATES:
                sdf[c] = hist[c].values
            xcols = COVARIATES
        models = {f"q{int(q*100)}": lgb.LGBMRegressor(objective="quantile", alpha=q,
                  n_estimators=200, num_leaves=63, verbosity=-1) for q in quantiles}
        ml = MLForecast(models=models, freq="h", lags=[1, 24, 168], date_features=[])
        ml.fit(sdf, static_features=[])
        if with_x and fut_cov is not None:
            Xf = fut_cov.reset_index(); Xf["unique_id"] = "z"
            fc = ml.predict(h=H, X_df=Xf[["unique_id", "ds"] + xcols])
        else:
            fc = ml.predict(h=H)
        cols = [f"q{int(q*100)}" for q in quantiles]
        return np.clip(fc[cols].values, 0, None)
    return fp

RESULTS.append(evaluate("LightGBM", lgb_fp_factory(False), False))
RESULTS.append(evaluate("LightGBM", lgb_fp_factory(True), True))

# %% [markdown]
# ## 6 · Foundation models — zero-shot
# %%
# --- Chronos-2 (Amazon) — native joint covariates ---
try:
    import chronos
except ModuleNotFoundError:
    import subprocess; subprocess.run(["uv", "pip", "install", "-q", "--system", "chronos-forecasting"])
try:
    from chronos import Chronos2Pipeline
    _chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)

    def chronos_fp_factory(with_x):
        def fp(hist, fut_cov, H, quantiles):
            target_vals = hist[TARGET].values.astype("float32")
            if with_x and fut_cov is not None:
                feats = [target_vals]
                for c in COVARIATES:
                    cov = np.concatenate([hist[c].values, fut_cov[c].values])
                    feats.append(cov.astype("float32")[:len(target_vals)])
                inp = torch.tensor(np.stack(feats, axis=0)).unsqueeze(0)
            else:
                inp = torch.tensor(target_vals).unsqueeze(0).unsqueeze(0)
            res = _chronos.predict_quantiles(inputs=inp, prediction_length=H,
                                             quantile_levels=quantiles)
            if isinstance(res, (list, tuple)):
                res = res[0]
            res = res.cpu().numpy() if hasattr(res, "cpu") else np.asarray(res)
            res = res[0, 0, :, :]   # (1, n_variates, H, n_quantiles) -> target variate 0
            return np.clip(res, 0, None)
        return fp
    RESULTS.append(evaluate("Chronos-2", chronos_fp_factory(False), False))
    RESULTS.append(evaluate("Chronos-2", chronos_fp_factory(True), True))
except Exception as e:
    print("Chronos-2 skipped:", e)

# %%
# --- TimesFM 2.5 (Google) ---
# forecast() returns [mean, q10..q90]; with return_backcast=True it PREPENDS the
# backcast, so slice the last H rows. P10/P50/P90 = cols [1,5,9].
try:
    import timesfm
except ModuleNotFoundError:
    import subprocess; subprocess.run(["uv", "pip", "install", "-q", "--system", "timesfm[xreg]"])
    import timesfm
try:
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    torch.set_float32_matmul_precision("high")
    _tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    _tfm.compile(timesfm.ForecastConfig(
        max_context=CONTEXT, max_horizon=H, normalize_inputs=True,
        use_continuous_quantile_head=True, infer_is_positive=True,
        fix_quantile_crossing=True, return_backcast=True))
    Q_IDX = [1, 5, 9]

    def timesfm_base(hist, fut_cov, H, quantiles):
        ctx = hist[TARGET].values.astype("float32")
        _, q = _tfm.forecast(horizon=H, inputs=[ctx])
        return np.clip(q[0][-H:][:, Q_IDX], 0, None)

    def timesfm_cov(hist, fut_cov, H, quantiles):
        ctx = hist[TARGET].values.astype("float32")
        dyn = {c: [np.concatenate([hist[c].values, fut_cov[c].values]).astype("float32")]
               for c in WEATHER}
        out, _ = _tfm.forecast_with_covariates(
            inputs=[ctx], dynamic_numerical_covariates=dyn, xreg_mode="timesfm + xreg")
        p50 = np.clip(np.asarray(out[0])[-H:], 0, None)
        _, qu = _tfm.forecast(horizon=H, inputs=[ctx]); qu = qu[0][-H:]
        return np.clip(np.column_stack([p50 + qu[:, 1] - qu[:, 5], p50,
                                        p50 + qu[:, 9] - qu[:, 5]]), 0, None)

    RESULTS.append(evaluate("TimesFM", timesfm_base, False))
    try:
        RESULTS.append(evaluate("TimesFM", timesfm_cov, True))
    except Exception as e:
        print("TimesFM covariate path skipped:", e)
except Exception as e:
    print("TimesFM skipped:", e)

# %% [markdown]
# > **Moirai is in a separate notebook** (`moirai_only.ipynb`). Its `uni2ts` dependency
# > pins **torch 2.4.1**, which is incompatible with the newer torch that Chronos-2 and
# > TimesFM 2.5 need — installing it here silently breaks the other foundation models.
# > Run that notebook on its own and fold its `RESULTS_JSON` into the table below.

# %%
# --- TabPFN-TS (Prior Labs) — known covariates, needs token ---
# Verified API (tabpfn-time-series 1.3.0): TabPFNTSPipeline.predict_df with
# context_df(item_id,timestamp,target,+covs) and future_df(item_id,timestamp,+covs);
# returns a frame with a `target` point column plus one column per quantile float.
if HAVE_TABPFN:
    try:
        from tabpfn_time_series import TabPFNTSPipeline
        _tabpfn_pipe = TabPFNTSPipeline()

        def tabpfn_fp_factory(with_x):
            def fp(hist, fut_cov, H, quantiles):
                feats = COVARIATES if (with_x and fut_cov is not None) else []
                ctx = pd.DataFrame({"item_id": "z", "timestamp": hist.index,
                                    "target": hist[TARGET].values})
                future_idx = (fut_cov.index if fut_cov is not None
                              else pd.date_range(hist.index[-1], periods=H + 1, freq="h")[1:])
                ftr = pd.DataFrame({"item_id": "z", "timestamp": future_idx})
                for c in feats:
                    ctx[c] = hist[c].values
                    ftr[c] = fut_cov[c].values
                pred = _tabpfn_pipe.predict_df(context_df=ctx, future_df=ftr,
                                               quantiles=list(quantiles))
                return np.clip(pred[list(quantiles)].values, 0, None)
            return fp
        RESULTS.append(evaluate("TabPFN-TS", tabpfn_fp_factory(False), False))
        RESULTS.append(evaluate("TabPFN-TS", tabpfn_fp_factory(True), True))
    except Exception as e:
        print("TabPFN-TS skipped:", e)
else:
    print("TabPFN-TS skipped — no token (see §1)")

# %% [markdown]
# ## 7 · Results
# %%
tbl = pd.DataFrame(RESULTS).sort_values("MASE").reset_index(drop=True)
print(tbl.to_string())

# %%
t = pd.DataFrame(RESULTS).sort_values("MASE")
fig, ax = plt.subplots(1, 2, figsize=(14, max(4, len(t) * 0.4)))
ax[0].barh(t["model"], t["MASE"]); ax[0].invert_yaxis()
ax[0].axvline(1.0, color="r", ls="--", lw=1, label="seasonal-naive")
ax[0].set_title("MASE (lower = better; <1 beats naive)"); ax[0].legend()
ax[1].barh(t["model"], t["pinball"]); ax[1].invert_yaxis()
ax[1].set_title("Pinball loss (probabilistic; lower = better)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Did covariates help?
# %%
r = pd.DataFrame(RESULTS)
r["base_model"] = r["model"].str.replace(" ＋cov", "", regex=False).str.replace(" base", "", regex=False)
piv = r.pivot_table(index="base_model", columns=r["model"].str.contains("cov"), values="MASE")
piv.columns = ["base", "＋cov"]
piv["Δ% (cov helps)"] = ((piv["base"] - piv["＋cov"]) / piv["base"] * 100).round(1)
print(piv.dropna().sort_values("Δ% (cov helps)", ascending=False).to_string())

# %% [markdown]
# ### 📋 Copy-paste results (send back to fill the blog post)
# %%
import json
print("RESULTS_JSON_START")
print(json.dumps({"h": H, "windows": N_WINDOWS, "results": RESULTS}, indent=2))
print("RESULTS_JSON_END")

# %% [markdown]
# ## 8 · 📝 Auto-generated blog draft
# %%
from datetime import date
_r = sorted(RESULTS, key=lambda x: x["MASE"])
best = _r[0]
def line(x): return f"| {x['model']} | {x['MAE']} | {x['MASE']} | {x['pinball']} | {x['coverage']} |"
post = f"""# Can Time Series Foundation Models Beat ARIMA & Prophet? Zurich Bike Counts

*{date.today():%B %Y} · {len(df):,} hourly observations · {H}h day-ahead · {N_WINDOWS}-window rolling backtest*

| Model | MAE | MASE | Pinball | Coverage |
|---|---|---|---|---|
{chr(10).join(line(x) for x in _r)}

**{best['model']}** led with MASE {best['MASE']} (MAE {best['MAE']} bikes/hour).
"""
print(post)
open("blog_post.md", "w").write(post)
print("\n— saved blog_post.md —")
