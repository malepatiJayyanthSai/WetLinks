"""
Extracts the bent-pipe RTT (dish -> satellite -> ground station -> Starlink gateway 100.64.0.1)
from net_traceroute.csv and merges it with the weather data from froggit.csv.

Output: one row per traceroute run -> Preprocessed_Data/bentpipe_<site>.csv

The script only extracts, labels and joins data. Nothing is removed or corrected here
(no outlier removal, no time-of-day correction) - that is left to the analysis.

Usage (from anywhere):
    python preprocessing/bentpipe_merger.py --site Osnabrück
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

BENTPIPE_HOP_ID = "100.64.0.1"  # Starlink gateway, first router after the radio segment

# site specific settings
SITE_CONFIG = {
    "Osnabrück": {
        # runs before this date used a dedicated test link (same cutoff as latency_analysis.py)
        "cutoff": pd.Timestamp("2023-09-14 15:12:00"),
    },
    "Enschede": {
        "cutoff": None,
    },
}

ROUTE_MIN_SHARE = 0.01        # a path needs >= 1% of runs to get its own route_id, rest -> "other"
WEATHER_HALF_WINDOW_MIN = 2   # rain_now etc. = mean over +-2 minutes around the run
WEATHER_COLS = ["temp", "humidity", "dewpt", "windspeed", "windgust", "solarradiation", "uv", "barom"]


def extract_ip(hop_id):
    """'name (1.2.3.4)' -> '1.2.3.4', '1.2.3.4' -> '1.2.3.4', '???' -> '???'"""
    match = re.search(r"\(([^)]+)\)\s*$", hop_id)
    return match.group(1) if match else hop_id.strip()


def load_runs(site):
    """Reads net_traceroute.csv and returns one row per traceroute run."""
    path = REPO_ROOT / "Raw_Data" / site / "net_traceroute.csv"
    tr = pd.read_csv(path)
    tr["timestamp_start"] = pd.to_datetime(tr["timestamp_start"])
    tr["timestamp_end"] = pd.to_datetime(tr["timestamp_end"])

    cutoff = SITE_CONFIG[site]["cutoff"]
    if cutoff is not None:
        tr = tr[tr["timestamp_start"] >= cutoff]

    # every run is identified by its start timestamp
    tr = tr.sort_values(["timestamp_start", "hop"]).reset_index(drop=True)
    tr["ip"] = tr["hop_id"].astype(str).map(extract_ip)

    # run level information
    grouped = tr.groupby("timestamp_start")
    runs = grouped.agg(
        timestamp_end=("timestamp_end", "first"),
        target=("target", "first"),
        hops=("hops", "max"),
        n_hop_rows=("hop", "size"),
        last_ip=("ip", "last"),
    )
    runs["reached_target"] = runs["last_ip"] == runs["target"]

    # bent-pipe hop (hop 1 in Osnabrück, hop 2 in Enschede -> select by address, not by number)
    bp = tr[tr["hop_id"] == BENTPIPE_HOP_ID].groupby("timestamp_start").first()
    bp = bp[["hop", "ping_avg", "ping_best", "ping_worst", "ping_stddev", "packet_loss", "packets_send"]]
    bp.columns = ["bp_hop", "bp_rtt_avg", "bp_rtt_best", "bp_rtt_worst", "bp_rtt_std", "bp_loss", "bp_packets_send"]
    runs = runs.join(bp)
    runs["bp_hop"] = runs["bp_hop"].astype("Int64")
    runs["bp_found"] = runs["bp_hop"].notna()

    # local gateway in front of the dish (only exists in Enschede, ~0.2 ms)
    gw = tr[tr["hop_id"].astype(str).str.startswith("_gateway")].groupby("timestamp_start")["ping_avg"].first()
    runs["gw_rtt"] = gw

    # path signature = IPs of the 3 hops after the bent-pipe hop (shows which Starlink route was used)
    tr = tr.join(runs["bp_hop"], on="timestamp_start")
    after_bp = tr[tr["hop"] > tr["bp_hop"]]
    path_sig = after_bp.groupby("timestamp_start").head(3).groupby("timestamp_start")["ip"].agg(">".join)
    runs["path_sig"] = path_sig

    # outage: no reply from the bent-pipe hop, complete loss on it, or a run that died after <= 2 hops
    runs["outage"] = (~runs["bp_found"]) | (runs["bp_loss"] >= 100) | (runs["hops"] <= 2)

    runs = runs.reset_index().rename(columns={"timestamp_start": "timestamp"})
    runs.insert(1, "site", site)
    return runs.drop(columns=["last_ip"])


def assign_route_ids(runs):
    """Common path signatures get R1, R2, ... in chronological order of first appearance, rare ones 'other'."""
    valid = runs[runs["bp_found"] & runs["path_sig"].notna()]
    share = valid["path_sig"].value_counts(normalize=True)
    common = share[share >= ROUTE_MIN_SHARE].index
    first_seen = valid[valid["path_sig"].isin(common)].groupby("path_sig")["timestamp"].min().sort_values()
    mapping = {sig: f"R{i + 1}" for i, sig in enumerate(first_seen.index)}
    runs["route_id"] = runs["path_sig"].map(mapping).fillna("other")
    return runs


def load_weather(site):
    """Reads froggit.csv and returns a regular 1-minute grid with the weather features."""
    path = REPO_ROOT / "Raw_Data" / site / "froggit.csv"
    wx = pd.read_csv(path)
    wx["timestamp"] = pd.to_datetime(wx["timestamp"])

    # when the outdoor sensor loses connection, the station reports -9999 codes
    # (-9999 °F = -5572.78 °C for temp/dewpt). rain is then reported as 0.0 although it was
    # not measured, so all outdoor values of such a reading are unknown. barom is measured
    # inside the console and stays valid.
    sensor_lost = (wx["temp"] < -100) | (wx["humidity"] == -9999)
    outdoor_cols = ["temp", "humidity", "dewpt", "windspeed", "windgust", "rain", "solarradiation", "uv"]
    wx.loc[sensor_lost, outdoor_cols] = np.nan

    # set remaining physically impossible values to NaN (instead of dropping the whole reading)
    for col in ["windspeed", "windgust", "rain", "solarradiation", "uv"]:
        wx.loc[wx[col] < 0, col] = np.nan
    wx.loc[(wx["humidity"] < 0) | (wx["humidity"] > 100), "humidity"] = np.nan

    # regular 1-minute grid; missing minutes stay NaN (no filling), duplicate readings are averaged
    grid = wx.set_index("timestamp")[["rain"] + WEATHER_COLS].resample("1min").mean()

    w = 2 * WEATHER_HALF_WINDOW_MIN + 1  # centred window, e.g. 5 minutes for +-2
    out = pd.DataFrame(index=grid.index)
    out["wx_readings"] = grid["rain"].notna().rolling(w, center=True, min_periods=1).sum()
    # note: froggit 'rain' is a rain RATE in mm/h
    out["rain_now"] = grid["rain"].rolling(w, center=True, min_periods=1).mean()
    out["rain_max_now"] = grid["rain"].rolling(w, center=True, min_periods=1).max()
    # rain in the minutes BEFORE the run (excluding the current minute)
    out["rain_prev_15"] = grid["rain"].rolling(15, min_periods=8).mean().shift(1)
    out["rain_prev_60"] = grid["rain"].rolling(60, min_periods=30).mean().shift(1)
    for col in WEATHER_COLS:
        out[col] = grid[col].rolling(w, center=True, min_periods=1).mean()

    # minutes since the last / until the next minute with rain (0 while it is raining)
    minute = pd.Series(grid.index, index=grid.index)
    rain_minute = minute.where(grid["rain"] > 0)
    out["min_since_rain"] = (minute - rain_minute.ffill()).dt.total_seconds() / 60
    out["min_until_rain"] = (rain_minute.bfill() - minute).dt.total_seconds() / 60
    return out


def merge_weather(runs, weather):
    """Attaches the weather of the minute containing the midpoint of each traceroute run."""
    midpoint = runs["timestamp"] + (runs["timestamp_end"] - runs["timestamp"]) / 2
    minute = midpoint.dt.floor("1min")
    wx = weather.reindex(minute.values)
    wx.index = runs.index
    runs = pd.concat([runs, wx], axis=1)
    runs["weather_ok"] = runs["wx_readings"].fillna(0) >= 1
    return runs.drop(columns=["wx_readings"])


def add_time_features(runs):
    ts = runs["timestamp"]
    runs["hour"] = ts.dt.hour
    runs["weekday"] = ts.dt.dayofweek  # 0 = Monday
    runs["is_weekend"] = runs["weekday"] >= 5
    runs["date"] = ts.dt.date
    runs["days_since_start"] = (ts - ts.min()).dt.total_seconds() / 86400
    return runs


def print_report(runs, weather):
    ok = runs[runs["bp_found"] & ~runs["outage"]]
    print("\n================ REPORT ================")
    print(f"runs:                {len(runs)}  ({runs['timestamp'].min()} .. {runs['timestamp'].max()})")
    print(f"bent-pipe hop found: {runs['bp_found'].mean() * 100:.2f}%   (bent-pipe hop number: {dict((int(k), int(v)) for k, v in runs['bp_hop'].value_counts().items())})")
    print(f"outages:             {runs['outage'].sum()}")
    print(f"reached target:      {runs['reached_target'].mean() * 100:.2f}%")
    print(f"weather_ok:          {runs['weather_ok'].mean() * 100:.2f}%")

    print("\nroutes:")
    routes = ok.groupby("route_id").agg(
        runs=("bp_rtt_avg", "size"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        median_bp_rtt=("bp_rtt_avg", "median"),
        path_sig=("path_sig", "first"),
    )
    print(routes.to_string())

    print("\nruns per rain bucket (rain_now, mm/h):")
    buckets = pd.cut(ok["rain_now"], bins=[-np.inf, 0, 1, 2, 4, np.inf], labels=["0", "0-1", "1-2", "2-4", ">4"])
    stats = ok.groupby(buckets, observed=False)["bp_rtt_avg"].agg(["size", "median"])
    stats.columns = ["runs", "median_bp_rtt"]
    print(stats.to_string())

    print("\nmedian bent-pipe RTT per hour:")
    per_hour = ok.groupby("hour")["bp_rtt_avg"].median()
    print("  " + "  ".join(f"{h}:{v:.1f}" for h, v in per_hour.items()))

    # time zone check: solar noon at ~7-8°E is ~11:30 UTC, so the sunniest hour bucket is
    # 11 for UTC timestamps, 12 for CET (winter local time), 13 for CEST (summer local time)
    solar = weather["solarradiation"].groupby(weather.index.hour).mean()
    print(f"\ntime zone check: mean solar radiation peaks in hour {solar.idxmax()} "
          f"(11 -> timestamps are UTC, 12 -> CET, 13 -> CEST)")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="Osnabrück", choices=list(SITE_CONFIG))
    args = parser.parse_args()

    print("reading traceroutes ...")
    runs = load_runs(args.site)
    runs = assign_route_ids(runs)
    print("reading weather ...")
    weather = load_weather(args.site)
    runs = merge_weather(runs, weather)
    runs = add_time_features(runs)

    columns = [
        "timestamp", "timestamp_end", "site", "route_id", "path_sig",
        "bp_rtt_avg", "bp_rtt_best", "bp_rtt_worst", "bp_rtt_std", "bp_loss", "bp_packets_send", "bp_hop", "gw_rtt",
        "hops", "n_hop_rows", "bp_found", "reached_target", "outage",
        "rain_now", "rain_max_now", "rain_prev_15", "rain_prev_60", "min_since_rain", "min_until_rain",
        *WEATHER_COLS, "weather_ok",
        "hour", "weekday", "is_weekend", "date", "days_since_start",
    ]
    runs = runs[columns]

    out_path = REPO_ROOT / "Preprocessed_Data" / f"bentpipe_{args.site}.csv"
    runs.to_csv(out_path, index=False)
    print(f"written {len(runs)} rows to {out_path}")

    print_report(runs, weather)


if __name__ == "__main__":
    main()
