"""'Like pitchers': how much do PITCHERS change by batter hand, vs how much HITTERS change by pitcher hand?
Pitchers (>= 300 pitches vs each batter side, per season): pitch-mix change (total variation distance
between mix vs RHB and vs LHB), sweeper/slider vs changeup shares by matchup, rubber shift (release x)."""

import glob, numpy as np, pandas as pd

df = pd.concat(
    [
        pd.read_parquet(f, columns=["pitcher", "game_year", "p_throws", "stand", "pitch_type", "release_pos_x"])
        for f in sorted(glob.glob("raw/*.parquet"))
    ],
    ignore_index=True,
).dropna(subset=["pitch_type"])
df["same"] = np.where(df.stand == df.p_throws, "same", "opp")
n = df.groupby(["pitcher", "game_year", "same"]).size().unstack()
keep = n[(n.same >= 300) & (n.opp >= 300)].index
d = df.set_index(["pitcher", "game_year"]).loc[keep].reset_index()
mix = d.groupby(["pitcher", "game_year", "same"]).pitch_type.value_counts(normalize=True).unstack(fill_value=0)
tvd = 0.5 * (mix.xs("same", level="same") - mix.xs("opp", level="same")).abs().sum(axis=1) * 100
print(f"pitcher-seasons: {len(tvd)}")
print(
    f"pitch-mix change by batter hand (share of pitches that switch type): median {tvd.median():.0f}%, "
    f"mean {tvd.mean():.0f}%, >= 10%: {(tvd >= 10).mean():.0%}, >= 20%: {(tvd >= 20).mean():.0%}"
)
for grp, pts in (("sweeper+slider", ["ST", "SL"]), ("changeup+splitter", ["CH", "FS"]), ("sinker", ["SI"])):
    sh = mix[[c for c in pts if c in mix.columns]].sum(axis=1).unstack("same") * 100
    print(f"  {grp:18s} vs same-hand {sh['same'].mean():5.1f}%   vs opposite-hand {sh['opp'].mean():5.1f}%")
rel = d.groupby(["pitcher", "game_year", "same"]).release_pos_x.mean().unstack()
shift = (rel["same"] - rel["opp"]).abs() * 12
print(
    f"release-point (rubber) shift by batter hand: median {shift.median():.1f} in, >= 2 in: {(shift >= 2).mean():.0%}"
)
h = pd.read_csv("hand_depth.csv")
print(
    f"\nHITTERS, by pitcher hand: depth diff |same - opp| median {h['diff'].abs().median():.1f} in, "
    f">= 2 in: {(h['diff'].abs() >= 2).mean():.0%}, >= 3 in: {(h['diff'].abs() >= 3).mean():.0%}"
)
