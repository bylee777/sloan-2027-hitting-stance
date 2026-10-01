"""Statcast per-pitch crawl for the batter's-box depth study (Sloan abstract).

Keeps the 9-parameter trajectory fit (vx0..az at y=50 ft), plate crossing, release,
and 2024+ bat-tracking columns (contact point vs batter). One parquet per half-month.
Resumable: existing files are skipped.
"""

import sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pybaseball as pb

RAW = Path(__file__).resolve().parent / "raw"
KEEP = [
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "player_name",
    "p_throws",
    "stand",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "arm_angle",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "description",
    "events",
    "type",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "delta_run_exp",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "hyper_speed",
    "miss_distance",
]


def windows(season):
    d, end = date(season, 3, 15), date(season, 10, 5)
    while d <= end:
        e = min(d + timedelta(days=14), end)
        yield d, e
        d = e + timedelta(days=1)


RAW.mkdir(exist_ok=True)
for season in [int(a) for a in sys.argv[1:]] or [2024, 2025, 2026]:
    for s, e in windows(season):
        out = RAW / f"{s:%Y%m%d}_{e:%Y%m%d}.parquet"
        if out.exists():
            continue
        for attempt, backoff in enumerate((0, 30, 90), 1):
            time.sleep(backoff)
            try:
                df = pb.statcast(start_dt=s.isoformat(), end_dt=e.isoformat(), verbose=False)
                break
            except Exception as exc:
                print(f"  {s}..{e} attempt {attempt} failed: {exc}", flush=True)
        else:
            raise SystemExit(f"gave up on {s}..{e}")
        if df is None or df.empty:
            print(f"{s}..{e}: 0 rows", flush=True)
            continue
        df = df[df["game_type"] == "R"]
        df[[c for c in KEEP if c in df.columns]].to_parquet(out, index=False)
        print(f"{s}..{e}: {len(df):,} pitches", flush=True)
print("DONE", flush=True)
