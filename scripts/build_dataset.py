#!/usr/bin/env python3
"""Build the Zurich bike/pedestrian + weather forecasting dataset.

Downloads two open-data feeds from Stadt Zürich, joins them into one hourly table
with a target count series and weather + calendar covariates, and saves a
self-contained parquet (the time-series analogue of the tabular project's
house_xy.npz).

Target      : hourly bicycle count at one counter (or the city-wide total).
Covariates  : weather (temperature, rain, humidity, radiation, wind) — past-observed
              calendar (hour, weekday, month, holiday) — future-known

Sources (CC-BY, Stadt Zürich open data):
  counts  : https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo
  weather : https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte

Usage:
  python scripts/build_dataset.py --years 2019 2020 2021 2022 2023
"""
import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

COUNTS_URL = ("https://data.stadt-zuerich.ch/dataset/"
              "ted_taz_verkehrszaehlungen_werte_fussgaenger_velo/download/"
              "{year}_verkehrszaehlungen_werte_fussgaenger_velo.csv")
WEATHER_URL = ("https://data.stadt-zuerich.ch/dataset/"
               "ugz_meteodaten_stundenmittelwerte/download/"
               "ugz_ogd_meteo_h1_{year}.csv")

# weather station that carries all parameters and sits centrally
WEATHER_STATION = "Zch_Stampfenbachstrasse"
# long-format Parameter codes -> friendly covariate names
WEATHER_PARAMS = {
    "T": "temp_c",          # air temperature °C
    "RainDur": "rain_min",  # rainfall duration, minutes per hour (the key driver)
    "Hr": "humidity_pct",   # relative humidity
    "StrGlo": "radiation",  # global radiation W/m²
    "WVv": "wind_ms",       # wind speed vector m/s
    "p": "pressure_hpa",    # air pressure
}


def _fetch(url):
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tfm-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def load_counts(years, counter_id):
    """Hourly bicycle counts (VELO_IN + VELO_OUT) for one counter, summed per hour."""
    frames = []
    for y in years:
        raw = _fetch(COUNTS_URL.format(year=y))
        df = pd.read_csv(io.BytesIO(raw))
        if counter_id is not None:
            df = df[df["FK_STANDORT"] == counter_id]
        df["ts"] = pd.to_datetime(df["DATUM"])
        df["velo"] = df[["VELO_IN", "VELO_OUT"]].sum(axis=1, min_count=1)
        # sum across counters (if aggregating) then resample 15min -> hourly
        hourly = (df.dropna(subset=["velo"])
                    .groupby("ts")["velo"].sum()
                    .resample("1h").sum(min_count=1))
        frames.append(hourly)
        print(f"  counts {y}: {hourly.notna().sum()} hourly obs", flush=True)
    return pd.concat(frames).sort_index().rename("velo_count")


def load_weather(years):
    """Hourly weather covariates, pivoted from long format, one central station."""
    frames = []
    for y in years:
        raw = _fetch(WEATHER_URL.format(year=y))
        df = pd.read_csv(io.BytesIO(raw))
        df = df[df["Standort"] == WEATHER_STATION]
        df = df[df["Parameter"].isin(WEATHER_PARAMS)]
        df["ts"] = pd.to_datetime(df["Datum"], utc=True).dt.tz_convert(None)
        wide = df.pivot_table(index="ts", columns="Parameter", values="Wert")
        wide = wide.rename(columns=WEATHER_PARAMS)
        frames.append(wide)
        print(f"  weather {y}: {len(wide)} hourly rows", flush=True)
    return pd.concat(frames).sort_index()


def add_calendar(df):
    """Future-known calendar covariates."""
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Covid regime flag (Swiss lockdowns: first 2020-03-16, broad restrictions into 2021)
    df["covid"] = ((idx >= "2020-03-16") & (idx < "2021-06-01")).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--counter", type=int, default=None,
                    help="FK_STANDORT counter id; default None = city-wide total")
    ap.add_argument("--out", default=os.path.join(DATA, "zurich_bikes.parquet"))
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    print(f"Years: {args.years} · counter: {args.counter or 'ALL (city total)'}")

    print("Downloading counts …")
    counts = load_counts(args.years, args.counter)
    print("Downloading weather …")
    weather = load_weather(args.years)

    print("Joining …")
    df = pd.DataFrame(counts).join(weather, how="inner")
    df = add_calendar(df)
    df = df.sort_index()

    # drop the leading/trailing rows where target is missing
    df = df[df["velo_count"].notna()]

    df.to_parquet(args.out)
    cov = [c for c in df.columns if c != "velo_count"]
    print(f"\n✅ wrote {args.out}")
    print(f"   {len(df):,} hourly rows · {df.index.min()} → {df.index.max()}")
    print(f"   target: velo_count (median {df['velo_count'].median():.0f}/h, "
          f"max {df['velo_count'].max():.0f})")
    print(f"   covariates ({len(cov)}): {', '.join(cov)}")
    miss = df.isna().mean().mul(100).round(1)
    if miss.max() > 0:
        print(f"   missing %: {miss[miss > 0].to_dict()}")


if __name__ == "__main__":
    main()
