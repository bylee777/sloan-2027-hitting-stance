"""Stance WIDTH (foot separation), 2024-26, clean zone.
A. Between hitters (pitch/PA level, pitcher-season FE + side/year): per 6 in WIDER -> whiff, chase, K, xwOBA-con,
   exit velo, bat speed, swing length. Matchup moderation with batter-season FE: width x same-hand, width x
   pitcher fastball velo (z), width x pitcher edge share (z).
B. Mid-season width changes (same detector as depth, park-adjusted, >= 4 in sustained): event-level change
   post[0,59] minus baseline[-90,-31], league-date adjusted; placebo = random dates on non-changers."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from fe import feols, demean
from stance_lib import pitches
import events as E

rng = np.random.default_rng(9)
s = pd.read_parquet("stance_adj.parquet").sort_values(["bsy", "game_date"])
s["width"] = s.width.fillna(s.groupby("bsy").width.transform("mean"))
codes = [pd.factorize(s.bsy)[0], pd.factorize(s.park)[0]]
resid = demean(s[["width"]].to_numpy(), codes)[:, 0]
s["width_adj"] = s.width - ((s.width - s.groupby("bsy").width.transform("mean")) - resid)
p = pitches()
p["bsy"] = p.batter.astype(str) + p.side + p.year
hs = (
    s.groupby("bsy")
    .agg(
        width=("width_adj", "mean"),
        depth=("depth_adj", "mean"),
        off_plate=("off_plate", "mean"),
        days=("depth", "size"),
    )
    .reset_index()
)
hs = hs[hs.days >= 20]
bio = pd.read_csv("bio_all.csv")[["batter", "height_in"]]
p = p.merge(hs, on="bsy").merge(bio, on="batter", how="left")
p["wide6"] = (p.width - p.width.mean()) / 6
p["deep_ft"] = (p.depth - p.depth.mean()) / 12
p["close_ft"] = -(p.off_plate - p.off_plate.mean()) / 12
p["h_z"] = (p.height_in - p.height_in.mean()) / p.height_in.std()
p["same"] = (p.stand == p.p_throws).astype(float)
p["psy"] = p.pitcher.astype(str) + p.year
p["count"] = p.balls.astype(str) + p.strikes.astype(str)
p["zone_i"] = p.zone.fillna(0).astype(int)
p["whiff_f"] = p.whiff.astype(float)
p["swing_f"] = p.swing.astype(float)
p["xw"] = p.estimated_woba_using_speedangle.astype(float)
p["ev"] = p.launch_speed.astype(float)
p["bs"] = p.bat_speed.astype(float)
p["sl"] = p.swing_length.astype(float)
p["sweet"] = p.launch_angle.between(8, 32).astype(float)
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
isPA = p.events.notna() & ~p.events.fillna("").astype(str).str.startswith(NON_PA)
p["K"] = np.where(isPA, p.events.isin(["strikeout", "strikeout_double_play"]) * 100.0, np.nan)
print(
    f"width: hitter-season mean {hs.width.mean():.1f} in, SD {hs.width.std():.1f}; corr with depth {hs.width.corr(hs.depth):+.2f}, off-plate {hs.width.corr(hs.off_plate):+.2f}"
)
OUT = (
    ("whiff per swing (pp)", lambda d: d[d.swing], "whiff_f", 100),
    ("chase (pp)", lambda d: d[d.out_zone], "swing_f", 100),
    ("K per PA (pp)", lambda d: d[isPA.loc[d.index]], "K", 1),
    ("xwOBA on contact (pts)", lambda d: d[(d.type == "X") & d.xw.notna()], "xw", 1000),
    ("exit velo (mph)", lambda d: d[(d.type == "X") & d.ev.notna()], "ev", 1),
    ("sweet-spot % BIP (pp)", lambda d: d[(d.type == "X") & d.launch_angle.notna()], "sweet", 100),
    ("bat speed (mph)", lambda d: d[d.bs.notna()], "bs", 1),
    ("swing length (ft)", lambda d: d[d.sl.notna()], "sl", 1),
)
print("\nA1. between hitters, per 6 in WIDER (pitcher-season FE; controls depth, plate distance, height, side, year)")
p["is_R"] = (p.side == "R").astype(float)
for lab, sel, y, sc in OUT:
    d = sel(p).dropna(subset=[y, "h_z"])
    b, V = feols(d, y, ["wide6", "deep_ft", "close_ft", "h_z", "is_R"], ["psy", "year"], "batter")
    print(f"   {lab:24s} {b['wide6']*sc:+7.2f} (z {b['wide6']/np.sqrt(V.loc['wide6','wide6']):+.1f})")
# matchup moderation: pitcher fastball velo and edge share (pitcher-season x batter side)
fbv = p[p.pitch_type.isin(["FF", "SI"])].groupby("psy").release_speed.mean().rename("fbv")
sgn = np.where(p.stand == "R", 1.0, -1.0)
xa = p.plate_x.astype(float) * sgn
p["edge"] = ((xa.abs() > 0.236) & (xa.abs() < 1.2)).astype(float)
p["pss"] = p.psy + p.stand
edge = p.groupby("pss").edge.mean().rename("edge_sh")
p = p.merge(fbv.reset_index(), on="psy", how="left").merge(edge.reset_index(), on="pss", how="left")
for c in ("fbv", "edge_sh"):
    p[c + "_z"] = (p[c] - p[c].mean()) / p[c].std()
p["w_same"] = p.wide6 * p.same
p["w_velo"] = p.wide6 * p.fbv_z
p["w_edge"] = p.wide6 * p.edge_sh_z
print(
    "\nA2. matchup moderation, per 6 in wider (batter-season FE + pitcher-season x side FE): x same-hand | x 1 SD pitcher FB velo | x 1 SD edge share"
)
for lab, sel, y, sc in OUT[:5]:
    d = sel(p).dropna(subset=[y, "fbv_z", "edge_sh_z"])
    b, V = feols(d, y, ["w_same", "w_velo", "w_edge"], ["bsy", "pss"], "batter")
    se = np.sqrt(np.diag(V))
    print(
        f"   {lab:24s} same {b['w_same']*sc:+.2f} (z {b['w_same']/se[0]:+.1f}) | velo {b['w_velo']*sc:+.2f} (z {b['w_velo']/se[1]:+.1f}) | edge {b['w_edge']*sc:+.2f} (z {b['w_edge']/se[2]:+.1f})"
    )

# ---------------- B. width events
ev = []
for _, g in s.groupby("bsy"):
    r = E.detect(g, "width", 4.0)
    if r:
        ev.append(r)
ev = pd.DataFrame(ev)
ev["dir"] = np.where(ev.move > 0, "WIDER", "NARROWER")
print(
    f"\nB. sustained mid-season width changes >= 4 in: {len(ev)} ({ev.dir.value_counts().to_dict()}), mean |move| {ev.move.abs().mean():.1f} in"
)
lg = {}
for lab, sel, y, sc in OUT:
    d = sel(p).dropna(subset=[y])
    m = d.groupby("game_date")[y].mean()
    d = d.assign(r=d[y] - d.game_date.map(m))
    lg[lab] = ({b: z for b, z in d[["bsy", "game_date", "r"]].groupby("bsy")}, sc)


def eff(lab, bsy, date):
    G, sc = lg[lab]
    z = G.get(bsy)
    if z is None:
        return np.nan
    t = (pd.to_datetime(z.game_date) - pd.Timestamp(date)).dt.days
    a, b = z.r[t.between(0, 59)], z.r[t.between(-90, -31)]
    return (a.mean() - b.mean()) * sc if len(a) >= 20 and len(b) >= 20 else np.nan


def runup(lab, bsy, date):
    G, sc = lg[lab]
    z = G.get(bsy)
    if z is None:
        return np.nan
    t = (pd.to_datetime(z.game_date) - pd.Timestamp(date)).dt.days
    a, b = z.r[t.between(-30, -1)], z.r[t.between(-90, -31)]
    return (a.mean() - b.mean()) * sc if len(a) >= 20 and len(b) >= 20 else np.nan


ctrl = [(b, g.game_date.to_numpy()) for b, g in s.groupby("bsy") if b not in set(ev.bsy) and len(g) >= 60]
pool = [(b, d[rng.integers(15, len(d) - 15)]) for b, d in ctrl for _ in range(2)]
for lab, *_ in OUT:
    pl = np.array([eff(lab, b, d) for b, d in pool])
    pl = pl[~np.isnan(pl)]
    cells = []
    for dr in ("WIDER", "NARROWER"):
        e = ev[ev.dir == dr]
        x = np.array([eff(lab, b, d) for b, d in zip(e.bsy, e.event_date)])
        x = x[~np.isnan(x)]
        pre = np.array([np.nan])
        null = np.array([rng.choice(pl, len(x), replace=False).mean() for _ in range(3000)])
        pv = (np.abs(null - null.mean()) >= abs(x.mean() - null.mean())).mean()
        ru = np.array([runup(lab, b, d) for b, d in zip(e.bsy, e.event_date)])
        ru = ru[~np.isnan(ru)]
        cells.append(
            f"{dr} {x.mean():+7.2f} (SE {x.std(ddof=1)/np.sqrt(len(x)):.2f}, p {pv:.3f}, n {len(x)}; run-up {ru.mean():+.2f} SE {ru.std(ddof=1)/np.sqrt(len(ru)):.2f})"
        )
    print(f"   {lab:24s} " + " | ".join(cells))
ev.to_csv("width_events.csv", index=False)
