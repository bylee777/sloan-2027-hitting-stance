"""How the posture data (Statcast zone top/bottom, 2024-25) relate to the paper's other models.
1. Zone rebuild (§3.3, used by chase / attack zones / accounting / intended speed / perception net): hitter-season median over ALL pitches
   vs over TAKES only (swings move the per-pitch values) — how different, and does the gap track swing rate?
2. Run-value accounting (§4.5): which parts posture maps onto, between hitters (2024-25, 250+ PA; height + year controls).
3. Matchups (§4.6): do hitters crouch differently against same- vs opposite-hand pitchers?
4. Stance-change events (§4.3): posture change that comes with width / depth changes (2024-25 events, 2 months after vs before).
5. Guerrero (§6): posture 2024 vs 2025."""

import glob, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
R = pd.concat(
    [
        pd.read_parquet(
            f, columns=["game_date", "game_type", "batter", "stand", "p_throws", "description", "sz_top", "sz_bot"]
        )
        for f in sorted(glob.glob("raw/2024*.parquet")) + sorted(glob.glob("raw/2025*.parquet"))
    ],
    ignore_index=True,
)
R = R[R.game_type == "R"].copy()
R["description"] = R.description.astype(object)
R["year"] = pd.to_datetime(R.game_date).dt.year.astype(str)
R["month"] = pd.to_datetime(R.game_date).dt.strftime("%Y-%m")
R["top"] = 12 * pd.to_numeric(R.sz_top, errors="coerce").astype(float)
R["bot"] = 12 * pd.to_numeric(R.sz_bot, errors="coerce").astype(float)
R["is_take"] = R.description.isin(["ball", "called_strike", "blocked_ball"])
R["is_swing"] = R.description.isin(
    ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt"]
)
R = R.rename(columns={"stand": "side"})
key = ["batter", "side", "year"]
H = pd.read_csv("bio_all.csv")[["batter", "height_in"]]
# ---- 1. zone rebuild: all pitches vs takes
A = (
    R.groupby(key)
    .agg(top_all=("top", "median"), bot_all=("bot", "median"), n=("top", "size"), swing_rate=("is_swing", "mean"))
    .reset_index()
)
T = R[R.is_take].groupby(key).agg(top_take=("top", "median"), bot_take=("bot", "median")).reset_index()
Z = A.merge(T, on=key)
Z = Z[Z.n >= 500]
Z["d_top"] = Z.top_all - Z.top_take
Z["d_bot"] = Z.bot_all - Z.bot_take
sw = R[R.is_swing].groupby(key).top.median().rename("top_swing").reset_index()
Z = Z.merge(sw, on=key)
print(
    f"1. ZONE REBUILD ({len(Z)} hitter-seasons, 500+ pitches): all-pitch median minus takes-only median: top {Z.d_top.mean():+.2f} in "
    f"(|d| mean {Z.d_top.abs().mean():.2f}, 90th pct {Z.d_top.abs().quantile(.9):.2f}); bottom {Z.d_bot.mean():+.2f} in (|d| {Z.d_bot.abs().mean():.2f})"
)
print(
    f"   swings vs takes, per-pitch median zone top: {(Z.top_swing - Z.top_take).mean():+.2f} in; corr(gap, swing rate) = {Z[['d_top', 'swing_rate']].corr().iloc[0, 1]:+.2f}"
)
# ---- 2. accounting parts vs posture (between hitters)
P = pd.read_csv("hitter_diagnosis_intended.csv")
P["year"] = P.year.astype(str)
P = P.merge(Z[key + ["top_take"]], on=key).merge(H, on="batter")
P = P[P.PA >= 250].copy()
P["crouch_pct"] = -100 * P.top_take / P.height_in  # +1 = zone top 1 point of height lower (more crouched)
print(
    f"\n2. RUN-VALUE ACCOUNTING vs posture ({len(P)} hitter-seasons, 250+ PA; per 1 point of height lower zone top; height + year controls; runs/600 PA)"
)
for part in ["DECISIONS", "WHIFFS", "INPUTS", "SQUARING", "LUCK", "OTHER", "TOTAL"]:
    r = smf.wls(f"{part} ~ crouch_pct + height_in + C(year)", P, weights=P.PA).fit(cov_type="HC1")
    print(f"   {part:10s} {r.params['crouch_pct']:+6.2f} (z {r.tvalues['crouch_pct']:+4.1f})")
# ---- 3. posture by pitcher hand (takes)
Hh = R[R.is_take].groupby(key + ["p_throws"]).agg(top=("top", "median"), n=("top", "size")).reset_index()
Hh = Hh[Hh.n >= 150].pivot_table(index=key, columns="p_throws", values="top").dropna().reset_index()
Hh["same_minus_opp"] = np.where(Hh.side == "R", Hh["R"] - Hh["L"], Hh["L"] - Hh["R"])
print(
    f"\n3. MATCHUPS: zone top vs same- minus opposite-hand pitchers, within hitter-season ({len(Hh)} with 150+ takes each): "
    f"mean {Hh.same_minus_opp.mean():+.2f} in, SD {Hh.same_minus_opp.std():.2f}; |diff| >= 1 in: {(Hh.same_minus_opp.abs() >= 1).mean():.1%}"
)
# ---- 4. posture change with stance-change events (2024-25)
Mo = R[R.is_take].groupby(["batter", "side", "month"]).agg(top=("top", "median"), n=("top", "size")).reset_index()
Mo = Mo[Mo.n >= 30]
Mo["t"] = pd.PeriodIndex(Mo.month, freq="M")
rows = []
for f, lever in (("width_events.csv", "width"), ("events_thr4.csv", "depth")):
    ev = pd.read_csv(f)
    ev = ev[ev.year.astype(str).isin(["2024", "2025"])]
    for _, e in ev.iterrows():
        h = Mo[(Mo.batter == e.batter) & (Mo.side == e.side)]
        if not len(h):
            continue
        k = (h.t - pd.Period(str(e.event_date)[:7], freq="M")).apply(lambda x: x.n)
        pre, post = h[k.between(-2, -1)], h[k.between(1, 2)]
        if len(pre) and len(post):
            rows.append(
                dict(
                    lever=lever,
                    size=e.persist,
                    d_top=np.average(post.top, weights=post.n) - np.average(pre.top, weights=pre.n),
                )
            )
E = pd.DataFrame(rows)
print(f"\n4. STANCE-CHANGE EVENTS (2024-25, 2 months after vs before): change in zone top (in)")
for lever in ("width", "depth"):
    e = E[E.lever == lever]
    r = smf.ols("d_top ~ I(size/3)", e).fit(cov_type="HC1")
    up, dn = e[e["size"] > 0].d_top, e[e["size"] < 0].d_top
    lab = ("wider", "narrower") if lever == "width" else ("deeper", "shallower")
    print(
        f"   {lever}: {lab[0]} {up.mean():+.2f} (n {len(up)}), {lab[1]} {dn.mean():+.2f} (n {len(dn)}); per 3 in of change {r.params.iloc[1]:+.2f} (z {r.tvalues.iloc[1]:+.1f})"
    )
# ---- 5. Guerrero
g = (
    R[(R.batter == 665489) & R.is_take]
    .groupby("year")
    .agg(top=("top", "median"), bot=("bot", "median"), n=("top", "size"))
)
print(
    "\n5. GUERRERO (takes): "
    + "; ".join(f"{y}: zone top {r.top:.1f} in, bottom {r.bot:.1f} in (n {r.n})" for y, r in g.iterrows())
)
