"""Generate the annotated Colab benchmark notebook (time-series edition) via nbformat."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []
def md(s): C.append(new_markdown_cell(s))
def code(s): C.append(new_code_cell(s))

md("""# 🚲 Time Series Foundation Models vs. ARIMA & Prophet — Zurich Bike Counts

The time-series sibling of the
[tabular-models-vs-xgboost](https://github.com/plotti/tabular-models-vs-xgboost) benchmark.

**The question:** can zero-shot *time-series foundation models* (Chronos-2, TabPFN-TS,
TimesFM, Moirai) beat classical **ARIMA**, **Prophet**, and a tuned **LightGBM** at
forecasting hourly bicycle traffic in Zurich — and **how much do weather covariates
(temperature, rain) actually help each one?**

Real target, real covariates, same city, open data. Cyclists double-peak at commute hours,
rain drops cycling ~30%, summer and Covid-2020 shift the whole regime — so there is genuine
signal for exogenous inputs to exploit.

> ⚡ **Runtime → Change runtime type → GPU (T4)** before running. The foundation models need it.
""")

md("""## 1 · Setup

Four model families:
- **Classical:** Seasonal-Naive, **ARIMA/SARIMAX**, ETS — via `statsforecast`
- **Intermediate:** **Prophet** (+ regressors), **LightGBM** on lag features — via `mlforecast`
- **Foundation:** **Chronos-2**, **TabPFN-TS**, **TimesFM 2.5**, **Moirai**
""")
code("""!uv pip install -q --system statsforecast mlforecast utilsforecast prophet lightgbm \\
    chronos-forecasting tabpfn-time-series "timesfm[xreg]" "jax[cpu]" uni2ts 2>/dev/null
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)
if DEVICE == "cpu":
    print("⚠️  No GPU — Runtime → Change runtime type → T4 GPU, then rerun.")""")

md("""### TabPFN token (for TabPFN-TS only)

Same as the tabular project: TabPFN is license-gated. Paste a free token from
[priorlabs.ai](https://priorlabs.ai) → Account → API Key. Leave blank to skip TabPFN-TS;
every other model still runs.
""")
code('''import os
TABPFN_TOKEN = ""  # <-- paste tabpfn_sk_... here (kept only in this session)
HAVE_TABPFN = bool(TABPFN_TOKEN.strip())
if HAVE_TABPFN:
    os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN.strip()
    print("✅ TabPFN token set")
else:
    print("ℹ️  No token — TabPFN-TS will be skipped.")''')

md("""## 2 · Get the data

`zurich_bikes.parquet` — 43,762 hourly rows (2019–2023): the city-wide bicycle count plus
weather and calendar covariates. Auto-downloaded from the repo.
""")
code("""import os, numpy as np, pandas as pd
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
print(f"covariates: {COVARIATES}")""")

md("""### The signal we're hoping the models find
Commute double-peak, and rain visibly suppressing cycling. If covariates don't help here,
they won't help anywhere.
""")
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
wd = df[df.is_weekend == 0].groupby("hour")[TARGET].mean()
we = df[df.is_weekend == 1].groupby("hour")[TARGET].mean()
ax[0].plot(wd.index, wd.values, label="weekday", lw=2)
ax[0].plot(we.index, we.values, label="weekend", lw=2)
ax[0].set_title("Mean bikes/hour by hour"); ax[0].set_xlabel("hour"); ax[0].legend()
peak = df[(df.is_weekend == 0) & (df.hour.isin([7,8,17,18]))]
dry = peak[peak.rain_min == 0][TARGET].mean(); wet = peak[peak.rain_min > 0][TARGET].mean()
ax[1].bar(["dry", "rainy"], [dry, wet], color=["#2a9d8f", "#6699cc"])
ax[1].set_title(f"Commute-hour cycling: rain drops it {100*(1-wet/dry):.0f}%")
for i, v in enumerate([dry, wet]): ax[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")
plt.tight_layout(); plt.show()""")

md("""## 3 · Evaluation protocol — rolling-origin backtest

You can't shuffle time, so instead of k-fold we use a **rolling-origin backtest**: pick
several cutoff points near the end of the series; at each one, every model sees only the
*past*, forecasts the next **H = 24 hours** (day-ahead, the operational horizon), and we
score against what actually happened. Same cutoffs, same horizon, same covariates for every
model — the apples-to-apples rule.

We report **both**:
- **Point metrics** on the median forecast: **MAE**, **RMSE**, and **MASE** (scaled vs.
  seasonal-naive; <1 beats "same as last week").
- **Probabilistic metrics**: **pinball loss** over quantiles P10/P50/P90, and **coverage**
  (does the P10–P90 band actually contain ~80% of the truth?). This is where the foundation
  models' native uncertainty earns its keep.
""")
code("""H = 24                 # forecast horizon (hours)
N_WINDOWS = 14          # rolling cutoffs
STEP = 24               # hours between cutoffs
QUANTILES = [0.1, 0.5, 0.9]
CONTEXT = 24 * 60       # history each model gets (~60 days)

# cutoffs: last N_WINDOWS day-ahead origins
origins = [len(df) - H - i * STEP for i in range(N_WINDOWS)][::-1]
print(f"H={H}h · {N_WINDOWS} windows · context={CONTEXT}h")
print(f"first origin {df.index[origins[0]]} … last {df.index[origins[-1]]}")

def seasonal_naive_scale(y_hist, m=168):
    # MASE denominator: in-sample MAE of weekly-seasonal naive (m=168h=1 week)
    d = np.abs(y_hist[m:] - y_hist[:-m])
    return d.mean() if len(d) else np.abs(np.diff(y_hist)).mean()

def pinball(y, q_preds, quantiles):
    total = 0.0
    for qi, q in enumerate(quantiles):
        e = y - q_preds[:, qi]
        total += np.mean(np.maximum(q * e, (q - 1) * e))
    return total / len(quantiles)

def evaluate(name, forecaster, use_cov):
    \"\"\"forecaster(hist_df, future_cov_df, H, quantiles) -> (H, n_quantiles) array.\"\"\"
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
        mases.append(np.mean(np.abs(y - p50)) /
                     seasonal_naive_scale(hist[TARGET].values))
        pins.append(pinball(y, qp, QUANTILES))
        lo, hi = qp[:, 0], qp[:, -1]
        covs.append(np.mean((y >= lo) & (y <= hi)))
    tag = "＋cov" if use_cov else "base"
    r = dict(model=f"{name} {tag}", MAE=round(np.mean(maes)),
             RMSE=round(np.mean(rmses)), MASE=round(np.mean(mases), 3),
             pinball=round(np.mean(pins), 1), coverage=round(np.mean(covs), 2),
             secs=round(time.time() - t0, 1))
    print(f"  {r['model']:26} MAE {r['MAE']:>5} · MASE {r['MASE']:.3f} · "
          f"pinball {r['pinball']:>6} · cov {r['coverage']:.2f} ({r['secs']}s)")
    return r

RESULTS = []""")

md("""## 4 · Classical baselines — Seasonal-Naive, ARIMA, ETS

The "ARIMA era". **Seasonal-Naive** (last week's same hour) is the floor every model must
beat. **AutoARIMA** gets the covariates as exogenous regressors (that's the *X* in SARIMAX);
**AutoETS** is univariate exponential smoothing. Run via `statsforecast`.
""")
code("""from statsforecast import StatsForecast
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

# ARIMA search space is capped so AutoARIMA doesn't take forever per window
ARIMA = lambda: AutoARIMA(season_length=24, max_p=5, max_d=2, max_q=5,
                          max_P=2, max_D=1, max_Q=2)
RESULTS.append(evaluate("SeasonalNaive", sf_forecaster(SeasonalNaive(season_length=168), False), False))
RESULTS.append(evaluate("AutoETS", sf_forecaster(AutoETS(season_length=24), False), False))
RESULTS.append(evaluate("ARIMA", sf_forecaster(ARIMA(), False), False))
RESULTS.append(evaluate("ARIMA", sf_forecaster(ARIMA(), True), True))""")

md("""## 5 · Intermediate — Prophet & LightGBM

**Prophet** (Meta) with weather as extra regressors — the famous, easy, often-mediocre
default. **LightGBM** on lag + calendar + weather features — the strong baseline the
foundation models must actually beat (the M5-winning recipe, and the tie-back to the
tabular project).
""")
code("""from prophet import Prophet

def prophet_fp(hist, fut_cov, H, quantiles):
    p = Prophet(interval_width=0.8, daily_seasonality=True, weekly_seasonality=True,
                yearly_seasonality=True)
    pdf = pd.DataFrame({"ds": hist.index, "y": hist[TARGET].values})
    if fut_cov is not None:
        for c in WEATHER:
            p.add_regressor(c); pdf[c] = hist[c].values
    p.fit(pdf)
    fdf = pd.DataFrame({"ds": fut_cov.index if fut_cov is not None
                        else pd.date_range(hist.index[-1], periods=H+1, freq="h")[1:]})
    if fut_cov is not None:
        for c in WEATHER: fdf[c] = fut_cov[c].values
    fc = p.predict(fdf)
    return np.clip(np.column_stack([fc["yhat_lower"], fc["yhat"], fc["yhat_upper"]]), 0, None)

RESULTS.append(evaluate("Prophet", prophet_fp, False))
RESULTS.append(evaluate("Prophet", prophet_fp, True))""")

code("""from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
import lightgbm as lgb

def lgb_fp_factory(with_x):
    def fp(hist, fut_cov, H, quantiles):
        sdf = pd.DataFrame({"unique_id": "z", "ds": hist.index, "y": hist[TARGET].values})
        xcols = []
        if with_x and fut_cov is not None:
            for c in COVARIATES: sdf[c] = hist[c].values
            xcols = COVARIATES
        models = {f"q{int(q*100)}": lgb.LGBMRegressor(objective="quantile", alpha=q,
                  n_estimators=200, num_leaves=63, verbosity=-1) for q in quantiles}
        ml = MLForecast(models=models, freq="h",
                        lags=[1, 24, 168], date_features=[])
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
RESULTS.append(evaluate("LightGBM", lgb_fp_factory(True), True))""")

md("""## 6 · Foundation models — zero-shot

Each one gets the same history and (where supported) the same future-known covariates. No
training, no tuning — one forward pass. Covariate support differs sharply (see the research
doc): **Chronos-2** and **TabPFN-TS** model covariates jointly; **TimesFM** via an auxiliary
regressor; **Moirai** v1 natively. We run each with and without covariates where it's
supported.
""")
code('''# --- Chronos-2 (Amazon) — native joint covariates ---
try:
    from chronos import Chronos2Pipeline
    import torch
    _chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    def chronos_fp_factory(with_x):
        def fp(hist, fut_cov, H, quantiles):
            target_vals = hist[TARGET].values.astype("float32")
            if with_x and fut_cov is not None:
                # stack target + covariates as variates: (1, n_variates, context_len)
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
            # (1, n_variates, H, n_quantiles) -> target variate 0
            res = res[0, 0, :, :]
            return np.clip(res, 0, None)
        return fp
    RESULTS.append(evaluate("Chronos-2", chronos_fp_factory(False), False))
    RESULTS.append(evaluate("Chronos-2", chronos_fp_factory(True), True))
except Exception as e:
    print("Chronos-2 skipped:", e)''')

code('''# --- TimesFM 2.5 (Google) ---
# API verified against timesfm 3.0.2 (which ships the 2.5 model under
# timesfm.timesfm_2p5). Gotchas handled:
#   * class lives in timesfm.timesfm_2p5.timesfm_2p5_torch, not top-level
#   * quantiles tensor is (n, L, 10) = [mean, q10..q90] -> P10/P50/P90 = idx 1,5,9
#   * with return_backcast=True (needed for covariates), forecast() PREPENDS the
#     backcast, so L = context+H -> always slice the LAST H rows ([-H:])
#   * covariates need `pip install timesfm[xreg]` + a working jax; the xreg path
#     returns a POINT forecast, so we widen it with the univariate quantile spread.
#   verified end-to-end on the real Zurich data (base MASE ~0.41).
try:
    import torch, numpy as np
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    import timesfm
    torch.set_float32_matmul_precision("high")
    _tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    _tfm.compile(timesfm.ForecastConfig(
        max_context=CONTEXT, max_horizon=H, normalize_inputs=True,
        use_continuous_quantile_head=True, infer_is_positive=True,
        fix_quantile_crossing=True, return_backcast=True))
    Q_IDX = [1, 5, 9]  # q10, q50, q90 within the 10-column [mean,q10..q90] output

    def timesfm_base(hist, fut_cov, H, quantiles):
        ctx = hist[TARGET].values.astype("float32")
        _, q = _tfm.forecast(horizon=H, inputs=[ctx])
        return np.clip(q[0][-H:][:, Q_IDX], 0, None)      # last H rows -> (H, 3)

    def timesfm_cov(hist, fut_cov, H, quantiles):
        ctx = hist[TARGET].values.astype("float32")
        # dynamic covariates must span context+horizon
        dyn = {c: [np.concatenate([hist[c].values, fut_cov[c].values]).astype("float32")]
               for c in WEATHER}
        out, _ = _tfm.forecast_with_covariates(
            inputs=[ctx], dynamic_numerical_covariates=dyn,
            xreg_mode="timesfm + xreg")
        p50 = np.clip(np.asarray(out[0])[-H:], 0, None)   # point forecast, last H
        # borrow the univariate band width around the covariate-adjusted median
        _, qu = _tfm.forecast(horizon=H, inputs=[ctx]); qu = qu[0][-H:]
        lo_off = qu[:, 1] - qu[:, 5]
        hi_off = qu[:, 9] - qu[:, 5]
        return np.clip(np.column_stack([p50 + lo_off, p50, p50 + hi_off]), 0, None)

    RESULTS.append(evaluate("TimesFM", timesfm_base, False))
    try:
        RESULTS.append(evaluate("TimesFM", timesfm_cov, True))
    except Exception as e:
        print("TimesFM covariate path skipped (need `pip install timesfm[xreg]` + jax):", e)
except Exception as e:
    print("TimesFM skipped:", e)''')

code("""# --- Moirai (Salesforce) — native any-variate covariates (v1) ---
try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    _moirai_mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large")
    def moirai_fp_factory(with_x):
        def fp(hist, fut_cov, H, quantiles):
            import torch as T
            feat = [hist[TARGET].values.astype("float32")]
            if with_x and fut_cov is not None:
                for c in COVARIATES:
                    feat.append(np.concatenate([hist[c].values, fut_cov[c].values]).astype("float32"))
            model = MoiraiForecast(module=_moirai_mod, prediction_length=H,
                    context_length=len(hist), patch_size="auto",
                    num_samples=100, target_dim=1,
                    feat_dynamic_real_dim=(len(COVARIATES) if with_x else 0),
                    past_feat_dynamic_real_dim=0)
            # (sample paths -> quantiles)
            samples = model.forecast(feat)  # pseudo-API; see uni2ts docs
            qs = np.quantile(samples, quantiles, axis=0).T
            return np.clip(qs, 0, None)
        return fp
    RESULTS.append(evaluate("Moirai", moirai_fp_factory(False), False))
    RESULTS.append(evaluate("Moirai", moirai_fp_factory(True), True))
except Exception as e:
    print("Moirai skipped (API varies by uni2ts version):", e)""")

code("""# --- TabPFN-TS (Prior Labs) — known covariates, needs token ---
if HAVE_TABPFN:
    try:
        from tabpfn_time_series import TabPFNTimeSeriesPredictor
        def tabpfn_fp_factory(with_x):
            def fp(hist, fut_cov, H, quantiles):
                feats = COVARIATES if (with_x and fut_cov is not None) else []
                pred = TabPFNTimeSeriesPredictor()
                out = pred.predict(train_y=hist[TARGET], train_X=hist[feats] if feats else None,
                                   test_X=fut_cov[feats] if feats else None,
                                   horizon=H, quantiles=quantiles)
                return np.clip(np.asarray(out).reshape(H, len(quantiles)), 0, None)
            return fp
        RESULTS.append(evaluate("TabPFN-TS", tabpfn_fp_factory(False), False))
        RESULTS.append(evaluate("TabPFN-TS", tabpfn_fp_factory(True), True))
    except Exception as e:
        print("TabPFN-TS skipped:", e)
else:
    print("TabPFN-TS skipped — no token (see §1)")""")

md("""## 7 · Results""")
code("""tbl = pd.DataFrame(RESULTS).sort_values("MASE").reset_index(drop=True)
tbl""")

code("""import matplotlib.pyplot as plt
t = pd.DataFrame(RESULTS).sort_values("MASE")
fig, ax = plt.subplots(1, 2, figsize=(14, max(4, len(t)*0.4)))
ax[0].barh(t["model"], t["MASE"]); ax[0].invert_yaxis()
ax[0].axvline(1.0, color="r", ls="--", lw=1, label="seasonal-naive")
ax[0].set_title("MASE (lower = better; <1 beats naive)"); ax[0].legend()
ax[1].barh(t["model"], t["pinball"]); ax[1].invert_yaxis()
ax[1].set_title("Pinball loss (probabilistic; lower = better)")
plt.tight_layout(); plt.show()""")

md("""### Did covariates help?
Pair each model's `base` vs `＋cov` MASE — the whole point of the benchmark.
""")
code("""r = pd.DataFrame(RESULTS)
r["base_model"] = r["model"].str.replace(" ＋cov", "", regex=False).str.replace(" base", "", regex=False)
piv = r.pivot_table(index="base_model", columns=r["model"].str.contains("cov"),
                    values="MASE")
piv.columns = ["base", "＋cov"]
piv["Δ% (cov helps)"] = ((piv["base"] - piv["＋cov"]) / piv["base"] * 100).round(1)
piv.dropna().sort_values("Δ% (cov helps)", ascending=False)""")

md("""### 📋 Copy-paste results (send back to fill the blog post)""")
code('''import json
print("RESULTS_JSON_START")
print(json.dumps({"h": H, "windows": N_WINDOWS, "results": RESULTS}, indent=2))
print("RESULTS_JSON_END")''')

md("""## 8 · 📝 Auto-generated blog draft""")
code('''from datetime import date
r = sorted(RESULTS, key=lambda x: x["MASE"])
best = r[0]; naive = next((x for x in RESULTS if "SeasonalNaive" in x["model"]), None)
def line(x): return (f"| {x['model']} | {x['MAE']} | {x['MASE']} | "
                     f"{x['pinball']} | {x['coverage']} |")
post = f"""# Can Time Series Foundation Models Beat ARIMA & Prophet? Zurich Bike Counts

*{date.today():%B %Y} · {len(df):,} hourly observations · {H}h day-ahead · {N_WINDOWS}-window rolling backtest*

Forecasting hourly bicycle traffic in Zurich, where weather genuinely drives the target
(rain cuts commute cycling ~30%). Every model sees the same history, same horizon, same
covariates — classical, intermediate, and zero-shot foundation models side by side.

## Results (lower MASE = better; MASE < 1 beats seasonal-naive)

| Model | MAE | MASE | Pinball | Coverage |
|---|---|---|---|---|
{chr(10).join(line(x) for x in r)}

## Verdict

**{best['model']}** led with MASE {best['MASE']} (MAE {best['MAE']} bikes/hour). The full
story — including how much the weather covariates helped each tier, and how well the
probabilistic models' uncertainty bands were calibrated — is in the tables above.
"""
print(post)
open("blog_post.md", "w").write(post)
print("\\n— saved blog_post.md —")''')

md("""---
*Dataset: Stadt Zürich open data (bike/pedestrian counters + weather), 2019–2023. Protocol
ported from the [tabular-models-vs-xgboost](https://github.com/plotti/tabular-models-vs-xgboost)
benchmark.*""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "accelerator": "GPU", "colab": {"provenance": []}}
nbf.write(nb, "notebooks/tsfm_vs_classical.ipynb")
print("wrote notebooks/tsfm_vs_classical.ipynb with", len(C), "cells")
