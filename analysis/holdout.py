"""Season-by-season replication of the paper's claims (each claim was found on pooled data; here every season
stands alone). Clean zone throughout.
C1 premise: hitters' depth vs same- minus opposite-hand pitchers (Savant hand splits), per season.
C2 depth x same-hand for chase-prone hitters (prior-season chase top third): chase + run value, 2025 vs 2026.
C3 depth movers (>= 4 in): post vs baseline chase / sweet-spot / xwOBA-con, by season of the move.
C4 width changes (>= 4 in): wider -> whiff, K, swing length; narrower -> bat speed, swing length; by season.
C5 plate-distance mechanism: per ft closer, whiff on AWAY / IN pitches vs middle, per season."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from fe import feols
from stance_lib import pitches

p = pitches()
p["bsy"] = p.batter.astype(str) + p.side + p.year
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
p["is_pa"] = p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)
p["K"] = np.where(p.is_pa, p.events.isin(["strikeout", "strikeout_double_play"]) * 100.0, np.nan)
p["whiff_f"] = p.whiff.astype(float)
p["swing_f"] = p.swing.astype(float)
p["xw"] = p.estimated_woba_using_speedangle.astype(float)
p["rv"] = p.delta_run_exp.astype(float) * 100
p["sweet"] = np.where(p.type == "X", p.launch_angle.between(8, 32) * 100.0, np.nan)
p["bs"] = p.bat_speed.astype(float)
p["sl"] = p.swing_length.astype(float)
p["same"] = (p.stand == p.p_throws).astype(float)

print("C1. depth vs SAME- minus OPPOSITE-hand pitchers (in), per season")
h = pd.read_csv("hand_depth.csv", dtype={"year": str})
for y, g in h.groupby("year"):
    print(
        f"   {y}: mean {g['diff'].mean():+.2f}, SD {g['diff'].std():.2f}, |diff| >= 3 in: {(g['diff'].abs() >= 3).sum()} of {len(g)}"
    )

print(
    "\nC2. chase-prone hitters, per ft deeper vs SAME-hand (same-minus-opposite): chase (- helps) | run value /100 (+ helps)"
)
prior = (
    p.groupby(["batter", "side", "year"])
    .apply(lambda d: (d.swing & d.out_zone).sum() / max(d.out_zone.sum(), 1))
    .rename("prior_chase")
    .reset_index()
)
prior["year"] = (prior.year.astype(int) + 1).astype(str)
s = pd.read_parquet("stance_adj.parquet")
seas = s.groupby("bsy").agg(depth=("depth_adj", "mean"), days=("depth", "size")).reset_index()
x = p[p.year.isin(["2025", "2026"])].merge(prior, on=["batter", "side", "year"]).merge(seas[seas.days >= 20], on="bsy")
x["cls"] = np.select(
    [
        x.pitch_type.isin(["FF", "SI", "FC"]),
        x.pitch_type.isin(["SL", "ST", "CU", "KC", "SV"]),
        x.pitch_type.isin(["CH", "FS"]),
    ],
    ["FB", "BRK", "OFF"],
    "X",
)
x = x[x.cls != "X"]
x["count"] = x.balls.astype(str) + x.strikes.astype(str)
x["zone_i"] = x.zone.fillna(0).astype(int)
x["dS_same"] = (x.depth - x.depth.mean()) / 12 * x.same
for y in ("2025", "2026"):
    xy = x[x.year == y]
    cut = xy.groupby("bsy").prior_chase.first().quantile(2 / 3)
    hi = xy[xy.prior_chase > cut]
    out = []
    for lab, d, col in (("chase", hi[hi.out_zone], "swing_f"), ("rv", hi.dropna(subset=["rv"]), "rv")):
        b, V = feols(d, col, ["dS_same"], ["bsy", ("same", "cls", "count"), ("same", "cls", "zone_i")], "batter")
        sc = 100 if lab == "chase" else 1
        out.append(f"{lab} {b['dS_same']*sc:+.2f} (z {b['dS_same']/np.sqrt(V.iloc[0,0]):+.1f})")
    print(f"   {y} (prior chase > {cut:.1%}, {hi.bsy.nunique()} hitters): " + " | ".join(out))


def events_by_season(ev, dirs, specs, title):
    print(f"\n{title}")
    for lab, mask, col, sc in specs:
        d = p[mask].dropna(subset=[col])
        m = d.groupby("game_date")[col].mean()
        d = d.assign(r=d[col] - d.game_date.map(m))
        G = {b: z for b, z in d[["bsy", "game_date", "r"]].groupby("bsy")}
        cells = []
        for yr in (2024, 2025, 2026):
            for dr in dirs:
                vals = []
                for _, e in ev[(ev.dir == dr) & (ev.year.astype(int) == yr)].iterrows():
                    z = G.get(e.bsy)
                    if z is None:
                        continue
                    t = (pd.to_datetime(z.game_date) - pd.Timestamp(e.event_date)).dt.days
                    a, c = z.r[t.between(0, 59)], z.r[t.between(-90, -31)]
                    if len(a) >= 20 and len(c) >= 20:
                        vals.append((a.mean() - c.mean()) * sc)
                v = np.array(vals)
                cells.append(
                    f"{yr} {dr[:4]} {v.mean():+.2f}±{v.std(ddof=1)/np.sqrt(len(v)):.2f} (n{len(v)})"
                    if len(v) > 1
                    else f"{yr} {dr[:4]} n{len(v)}"
                )
        print(f"   {lab:14s} " + " | ".join(cells))


ev_d = pd.read_csv("events_thr4.csv")
events_by_season(
    ev_d,
    ("BACK",),
    (
        ("chase pp", p.out_zone, "swing_f", 100),
        ("sweet-spot pp", p.sweet.notna(), "sweet", 1),
        ("xwOBA-con", (p.type == "X") & p.xw.notna(), "xw", 1000),
    ),
    "C3. DEPTH movers back: post minus baseline, by season (mean ± SE)",
)
ev_w = pd.read_csv("width_events.csv")
events_by_season(
    ev_w,
    ("WIDER", "NARROWER"),
    (
        ("whiff pp", p.swing, "whiff_f", 100),
        ("K pp", p.is_pa, "K", 1),
        ("swing len ft", p.sl.notna(), "sl", 1),
        ("bat speed", p.bs.notna(), "bs", 1),
    ),
    "C4. WIDTH changes: post minus baseline, by season",
)

print("\nC5. per ft CLOSER to the plate: whiff on AWAY / IN pitches vs middle, per season")
hs = pd.read_csv("hitter_traits.csv", dtype={"bsy": str})[["bsy", "close_ft"]]
q = p.merge(hs, on="bsy")
q = q[q.swing]
sgn = np.where(q.stand == "R", 1.0, -1.0)
xa = q.plate_x.astype(float) * sgn
q["cls"] = np.select(
    [
        q.pitch_type.isin(["FF", "SI", "FC"]),
        q.pitch_type.isin(["SL", "ST", "CU", "KC", "SV"]),
        q.pitch_type.isin(["CH", "FS"]),
    ],
    ["FB", "BRK", "OFF"],
    "X",
)
q["count"] = q.balls.astype(str) + q.strikes.astype(str)
q["zone_i"] = q.zone.fillna(0).astype(int)
q["c_away"] = q.close_ft * (xa > 0.236)
q["c_in"] = q.close_ft * (xa < -0.236)
for y in ("2024", "2025", "2026"):
    d = q[q.year == y]
    b, V = feols(d, "whiff_f", ["c_away", "c_in"], ["bsy", ("cls", "count"), ("cls", "zone_i")], "batter")
    se = np.sqrt(np.diag(V))
    print(
        f"   {y}: AWAY {b['c_away']*100:+.2f} (z {b['c_away']/se[0]:+.1f}) | IN {b['c_in']*100:+.2f} (z {b['c_in']/se[1]:+.1f})"
    )
