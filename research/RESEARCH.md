# Time Series Foundation Models — Research & Project Plan

*The time-series sibling of the [tabular-models-vs-xgboost](https://github.com/plotti/tabular-models-vs-xgboost)
benchmark. Same idea: take a real, messy dataset where **multiple inputs drive a target
series**, and shoot classical baselines → intermediate models → foundation models at it
under one identical evaluation protocol.*

Date: October 2026

---

## 1. The core design constraint: covariates

Your requirement — *"a time series where multiple inputs influence the target"* — is
exactly the hard part of this field, and it decides everything downstream.

Most time-series foundation models (TSFMs) are **univariate**: they only look at the
target's own past. That's useless for the question we actually care about ("does weather /
load / promotions improve the forecast?"). So the whole project has to be built around
**exogenous covariates**, and that narrows both the dataset choice and which foundation
models can even compete.

**Covariate support across the 2026 foundation models** (this is the key table):

| Model | Covariate support | Notes |
|---|---|---|
| **Chronos-2** (Amazon, Oct 2025) | ✅ **Native, joint** | Time + group attention; models target & covariates together. Most production-ready. |
| **TabPFN-TS** (Prior Labs) | ✅ Known covariates | The tabular model from our last project, adapted to TS. Nice continuity. No past-only covariates / multivariate targets. |
| **TimesFM 2.5** (Google) | ⚠️ Auxiliary only | Univariate backbone + a separate linear regressor on covariates at inference. Does *not* ingest known/categorical covariates in the transformer. |
| **Moirai 1.x** (Salesforce) | ✅ Native (any-variate) | The original covariate pioneer. |
| **Moirai 2.0** (Salesforce) | ❌ Univariate | Traded covariate support for better univariate scores. |
| **Chronos (v1)** | ❌ Univariate | Token-quantization T5. |
| **TiRex, Moment, Sundial, Toto** | ❌ mostly univariate | — |

**Implication:** the headline comparison is **Chronos-2 and TabPFN-TS** (covariate-aware
foundation models) vs. the classical/intermediate baselines that *also* use covariates.
TimesFM and a univariate foundation model should be included as an honest "univariate
ceiling" — it shows how much the exogenous inputs actually buy you.

---

## 2. Dataset candidates

Ranked for our purpose: a genuinely multivariate target with real exogenous drivers,
publicly downloadable, complex enough to be interesting but not a toy.

### 🥇 Recommended: EPF — Electricity Price Forecasting

- **What:** Hourly day-ahead electricity price for 5 markets (Nord Pool, PJM, Belgium,
  France, Germany), 6 years each.
- **Target:** hourly day-ahead price.
- **Exogenous (future-known!):** next-day load forecast + wind/generation forecast per
  market. These are *known in advance*, which is the ideal covariate setting.
- **Why it wins:** purpose-built for forecasting-with-covariates (it's the benchmark from
  the TimeXer paper). Strong daily/weekly seasonality, price spikes, regime shifts driven
  by the demand/renewables balance — so covariates genuinely matter. Clean target vs.
  exogenous split. Realistic business problem.
- **Source:** `epftoolbox` (open Python library, auto-downloads the 5 datasets).

### 🥈 Strong alternative: ETT — Electricity Transformer Temperature

- **What:** 2 years of transformer sensor data from China, hourly (ETTh1/h2) and
  15-min (ETTm1/m2).
- **Target:** oil temperature. **Exogenous:** 6 power-load features (HUFL, HULL, MUFL,
  MULL, LUFL, LULL).
- **Why:** the cleanest endogenous/exogenous textbook split, tiny and fast to iterate on,
  universally used so results are comparable. Downside: covariates are *past-only* (not
  future-known), and it's a bit of an ML-benchmark cliché.

### 🥉 Richest / most "real business": M5 (Walmart retail)

- **What:** 42k daily sales series, Walmart, hierarchical (item/store/state).
- **Target:** daily unit sales. **Exogenous:** price, promotions, SNAP days, calendar
  events, weekday — a mix of future-known and past-only.
- **Why:** the gold standard for covariate-rich demand forecasting; LightGBM won it with
  hand-crafted features (perfect baseline story). **Downside:** intermittent/zero-heavy
  demand breaks many models, and the hierarchy adds complexity. Great as a stretch goal,
  heavy as a starting point.

### Other solid options

- **Beijing PM2.5** — air quality (target) driven by weather covariates (dew point, temp,
  pressure, wind, snow, rain). Very intuitive "multiple inputs drive one series" story.
- **Max Planck Weather** — 21 meteo variables, one target; classic but covariates are
  past-only.
- **Chilean grid energy** — thermal generation target, renewables+storage as covariates;
  strong cross-source coupling, newer and less overfit.

### Recommendation

**Start with EPF.** It's the sweet spot: real problem, genuinely future-known covariates
(so the covariate story is clean), manageable size, and a mature download path. Keep
**ETT** as the quick smoke-test dataset, and hold **M5** as the ambitious follow-up once
the harness works.

---

## 3. The baseline ladder (your question)

Mirroring the tabular project's "classical → intermediate → foundation" structure, here is
the recommended line-up. Your instinct (ARIMA + Prophet as intermediary + foundation
models) is right; here's the fleshed-out version.

### Tier 1 — Classical statistics (the "ARIMA era")
- **Seasonal Naive** — the honest floor. Forecast = last week's same hour. Every model
  must beat this or it's worthless.
- **(S)ARIMA / SARIMAX** — Box-Jenkins. `SARIMAX` is the key one: it's ARIMA **with
  exogenous regressors**, so it actually uses the covariates. This is your "ARIMA baseline".
- **ETS / Holt-Winters** — exponential smoothing, univariate. Fast, explainable, no
  covariates — a good "how far does pure seasonality get you" marker.
- *(via `statsforecast` for speed — it runs these at scale in Rust/Numba.)*

### Tier 2 — Intermediate / ML (the "global model era")
- **Prophet** (Meta) — your requested intermediary. Additive trend+seasonality+holidays,
  supports extra regressors. Famous, easy, often mediocre — exactly the honest "popular
  default" data point.
- **LightGBM / XGBoost on lag features** — the M5-winning recipe: hand-built lags, rolling
  stats, calendar + covariates fed to gradient boosting. This is usually the *strong*
  baseline that foundation models have to beat, and it ties directly back to the tabular
  project.
- *(optional: a modern supervised deep model like TFT or TiDE/PatchTST — the "deep learning
  era" — if we want a complete arc. Can be v2.)*

### Tier 3 — Foundation models (zero-shot)
- **Chronos-2** (Amazon) — primary covariate-aware foundation model. Native joint
  target+covariate attention. The one to beat.
- **TabPFN-TS** (Prior Labs) — covariate-aware, and a direct callback to our tabular
  benchmark. (Also lets us re-use the TABPFN token story + licensing angle.)
- **TimesFM 2.5** (Google) — covariates via auxiliary regressor; the "fast univariate
  backbone" data point.
- **Moirai** (Salesforce) — include v1 for native covariates and/or v2.0 as the univariate
  reference.

### The key experiment
Run each foundation model **with and without covariates** where possible. The headline
isn't just "foundation model vs. ARIMA" — it's **"how much do the exogenous inputs actually
help each tier?"** That's the time-series analogue of what the energy-class feature did in
the tabular project, and it's the genuinely interesting result.

---

## 4. Evaluation protocol (ported from the tabular project)

- **Rolling-origin / expanding-window backtest** (the time-series equivalent of k-fold —
  you can't shuffle time). Multiple cutoffs, forecast a fixed horizon each time.
- **Horizon:** for EPF, 24h day-ahead (the operational horizon). For ETT, 96/192/336/720
  steps (the standard long-horizon splits).
- **Metrics:** MAE, RMSE, **MASE** (scaled against seasonal naive — the fair cross-series
  metric), **sMAPE**, and a **quantile/pinball loss** for probabilistic models (foundation
  models are probabilistic by default — we should use that, not just point forecasts).
- **Same splits, same horizon, same covariates for every model** — the apples-to-apples
  rule that made the tabular post credible.
- Watch for **data leakage**: TSFM benchmarks are notoriously contaminated (pretraining
  corpora overlap with public test sets — documented 47–184% MSE advantage on leaked data).
  Note which datasets each model likely saw; EPF/ETT are common pretraining sets, so flag it.

---

## 5. Tooling

- **`nixtla` ecosystem** — `statsforecast` (ARIMA/ETS/naive, fast), `mlforecast`
  (LightGBM+lags), `neuralforecast` (TFT/PatchTST), `utilsforecast` (metrics). One
  consistent API across tiers. This is the backbone.
- **Prophet** — `prophet`.
- **Chronos-2** — `chronos-forecasting` (Amazon, pip).
- **TabPFN-TS** — `tabpfn-time-series` (needs the Prior Labs token, like last project).
- **TimesFM** — `timesfm` (Google).
- **Moirai** — `uni2ts` (Salesforce).
- **Data:** `epftoolbox` for EPF; ETT/M5 are direct CSV downloads.
- GPU needed for the foundation models (Colab T4), exactly like the tabular notebook.

---

## 6. Proposed deliverables (same shape as the tabular project)

1. `scripts/build_dataset.py` — download EPF, assemble target + covariate matrix, save a
   self-contained `.npz`/parquet (the `house_xy.npz` analogue).
2. `notebooks/tsfm_vs_classical.ipynb` — annotated Colab notebook: all tiers through one
   rolling-backtest harness, with/without covariates, charts + leaderboard, auto-generated
   blog draft.
3. A public GitHub repo + a blog post ("How good are Time Series Foundation Models — can
   Chronos-2 beat ARIMA and Prophet when covariates actually matter?").

---

## 7. Decisions (confirmed)

1. **Dataset: ✅ Zurich bike & pedestrian counters + Zurich weather** (see §8). Chosen over
   EPF/ETT/M5 for the hands-on story: real local target, real covariate, same city, open
   license, and vivid covariate effects (commute double-peaks, rain dips, summer holidays,
   Covid 2020, snow days).
2. **Foundation models: ✅ all four** — Chronos-2, TabPFN-TS, TimesFM 2.5, Moirai.
3. **Point vs. probabilistic: ✅ both.** Point metrics (MAE, RMSE, MASE, sMAPE) as the
   headline everyone understands, **plus** pinball/quantile loss (P10/P50/P90) as the
   "calibrated uncertainty, for free" section — the probabilistic foundation models' real
   strength. Report coverage (does the P10–P90 band actually contain ~80% of actuals?) too.

## 8. Chosen dataset: Zurich bikes/pedestrians + weather

Open data from Stadt Zürich (city of Zurich), CC-BY style open licence.

**Target — bike & pedestrian counts**
- Dataset: https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo
- One CSV per year, e.g. `…/download/2023_verkehrszaehlungen_werte_fussgaenger_velo.csv`
- **15-minute** values; automatic counters across the city since 2009.
- Pick one well-behaved counter location (or aggregate the city total) as the target series.

**Covariates — weather (join on time)**
- Dataset: https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte
- Files `ugz_ogd_meteo_h1_YYYY.csv`; **hourly** values 1992→now (yearly CSVs + one full parquet).
- Key exogenous drivers: **temperature, precipitation/rain**, plus humidity, wind,
  radiation. These are *past-observed*; for a future-known covariate setup use a weather
  *forecast* feed or treat calendar features (hour, weekday, holiday, school holidays) as
  the known-future covariates.

**Resampling note:** counts are 15-min, weather is hourly → aggregate counts to **hourly**
to join cleanly (sum counts per hour, align to the weather timestamp). Hourly is also the
natural operational resolution.

**Why it's a good test:** daily commute double-peaks (7–9h, 17–19h), strong weekday/weekend
split, rain suppresses cycling sharply, summer-holiday and Covid-2020 regime shifts — so
temperature/rain covariates should *measurably* help, which is exactly the hypothesis the
benchmark is built to test.

**Feature ideas**
- Target: hourly count at a chosen counter (or city aggregate).
- Past-observed covariates: temperature, rainfall, humidity, wind, global radiation.
- Future-known covariates: hour-of-day, day-of-week, month, public holiday, school holiday
  (Canton Zürich), maybe a Covid-lockdown flag.

### ✅ Built (Oct 2026) — `data/zurich_bikes.parquet`

`scripts/build_dataset.py` downloads counts + weather, resamples to hourly, joins, and
saves the self-contained parquet. Result:

- **43,762 hourly rows**, 2019-01-01 → 2023-12-31 (5 years, incl. Covid regime).
- Target `velo_count` = city-wide hourly bicycle total (VELO_IN+OUT across all counters).
  Median 931/h, max 7,461/h.
- 11 covariates: weather (`temp_c`, `rain_min`, `humidity_pct`, `radiation`, `wind_ms`,
  `pressure_hpa`) + calendar (`hour`, `dow`, `month`, `is_weekend`, `covid`).
- <1% missing. Weather from station Zch_Stampfenbachstrasse (long→wide pivot).

**Covariate signal validated** (the whole premise — these really drive the target):
- Commute double-peak: 8h ≈ 3,690/h and 17–18h ≈ 3,850/h on weekdays.
- **Rain drops weekday-commute cycling ~30%** (3,732/h dry → 2,621/h rainy).
- corr(count, temp) = **0.46**.

So the "do covariates help each model tier?" experiment has real signal to find.

---

## Sources

- [The Forecasting Company — From ARIMA to Foundation Models](https://www.theforecastingcompany.com/blog/from-arima-to-foundation-models/)
- [Chronos-2: From Univariate to Universal Forecasting (arXiv 2510.15821)](https://arxiv.org/pdf/2510.15821)
- [TimeXer: Transformers for Forecasting with Exogenous Variables (arXiv 2402.19072)](https://arxiv.org/html/2402.19072)
- [Investigating target–covariate relationships for Chronos-2 and TabPFN-TS (arXiv 2605.12200)](https://arxiv.org/pdf/2605.12200)
- [Challenges and Requirements for Benchmarking TSFMs (arXiv 2510.13654)](https://arxiv.org/html/2510.13654v2)
- [The 2026 Time Series Toolkit: 5 Foundation Models (MachineLearningMastery)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [GIFT-Eval benchmark (Salesforce)](https://github.com/SalesforceAIResearch/gift-eval)
- [epftoolbox — Electricity Price Forecasting datasets](https://github.com/jeslago/epftoolbox)
