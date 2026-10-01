"""Do hitters change stance when SLUMPING, and do the changes work beyond regression to the mean (bounce-back)?
Windows around a reference date r: baseline B [r-90, r-31], run-up U [r-30, r-1], after A [r, r+59] (calendar days).
Changers: 201 width + 46 depth events. Controls: every 3rd day of every hitter-season with no event within +-90 days.
(1) Trigger: run-up minus baseline (the slump) for changers vs controls; logistic hazard of changing.
(2) Effect beyond bounce-back: each changer matched to the 20 nearest controls (same season, reference date within
    +-30 days) on slump size (xwOBA, whiff) and baseline xwOBA; effect = changer's (after - run-up) minus matched
    controls' (after - run-up). Compare with the naive before/after."""

import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
D = pd.read_parquet("hitter_day_outcomes.parquet")
D["t"] = (pd.to_datetime(D.game_date) - pd.Timestamp("2024-01-01")).dt.days
COLS = ["PA", "xw_sum", "K", "sw", "wh", "oz", "oz_sw", "bip", "sweet", "bs_sum", "bs_n"]
MET = {
    "xwoba": ("xw_sum", "PA", 1000),
    "whiff": ("wh", "sw", 100),
    "K": ("K", "PA", 100),
    "chase": ("oz_sw", "oz", 100),
    "sweet": ("sweet", "bip", 100),
    "bat_speed": ("bs_sum", "bs_n", 1),
}
W = pd.read_csv("width_events.csv")
W["lever"] = "width"
E = pd.read_csv("events_thr4.csv")
E["lever"] = "depth"
EV = pd.concat([W, E], ignore_index=True)
EV["year"] = EV.year.astype(str)
EV["t"] = (pd.to_datetime(EV.event_date) - pd.Timestamp("2024-01-01")).dt.days
rows = []
for (b, s, y), g in D.groupby(["batter", "side", "year"]):
    g = g.sort_values("t")
    tt = g.t.to_numpy()
    cs = np.vstack([np.zeros(len(COLS)), np.cumsum(g[COLS].to_numpy(float), 0)])

    def win(lo, hi):
        i, j = np.searchsorted(tt, lo, "left"), np.searchsorted(tt, hi, "right")
        return cs[j] - cs[i]

    evs = EV[(EV.batter == b) & (EV.side == s) & (EV.year == y)]
    refs = [(int(e.t), f"{e.lever}:{e.dir}", e["name"]) for _, e in evs.iterrows()]
    evt = evs.t.to_numpy()
    for r in range(int(tt.min()) + 90, int(tt.max()) - 59, 3):
        if len(evt) == 0 or np.abs(evt - r).min() > 90:
            refs.append((r, "control", None))
    for r, kind, nm in refs:
        B, U, A = win(r - 90, r - 31), win(r - 30, r - 1), win(r, r + 59)
        if min(B[0], U[0]) < 40 or A[0] < 60:
            continue
        row = dict(batter=b, side=s, year=y, r=r, kind=kind, name=nm)
        for m, (n, d, sc) in MET.items():
            i, j = COLS.index(n), COLS.index(d)
            for lab, v in (("B", B), ("U", U), ("A", A)):
                row[f"{m}_{lab}"] = sc * v[i] / v[j] if v[j] > 0 else np.nan
        rows.append(row)
X = pd.DataFrame(rows)
X["slump_xw"] = X.xwoba_U - X.xwoba_B
X["slump_wh"] = X.whiff_U - X.whiff_B
ch, co = X[X.kind != "control"], X[X.kind == "control"]
print(f"changers with full windows: {len(ch)} of {len(EV)}; control reference dates: {len(co):,}")
print("\n(1) TRIGGER — run-up minus baseline (the 30 days before the change vs 31-90 days before):")
for m in ("xwoba", "whiff", "K", "chase", "bat_speed"):
    a, c = ch[f"{m}_U"] - ch[f"{m}_B"], co[f"{m}_U"] - co[f"{m}_B"]
    print(
        f"   {m:9s} changers {a.mean():+6.2f} (SE {a.std()/np.sqrt(a.notna().sum()):.2f}) vs controls {c.mean():+6.2f}"
    )
X["changed"] = (X.kind != "control").astype(int)
lg = smf.logit("changed ~ I(slump_xw/10) + I(slump_wh) + I(xwoba_B/10) + C(year)", X).fit(disp=0)
print(
    f"   hazard of changing: per 10 pts xwOBA drop in the run-up odds x{np.exp(-lg.params['I(slump_xw / 10)']):.2f} (z {-lg.tvalues['I(slump_xw / 10)']:+.1f}); "
    f"per +1 pp whiff odds x{np.exp(lg.params['I(slump_wh)']):.2f} (z {lg.tvalues['I(slump_wh)']:+.1f})"
)
# (2) matched comparison
Z = ["slump_xw", "slump_wh", "xwoba_B"]
sd = co[Z].std()
res = []
for _, e in ch.iterrows():
    pool = co[(co.year == e.year) & ((co.r - e.r).abs() <= 30) & (co.batter != e.batter)].dropna(subset=Z)
    dist = (((pool[Z].astype(float) - e[Z].astype(float)) / sd) ** 2).sum(1).astype(float)
    m = pool.loc[dist.nsmallest(20).index]
    row = dict(kind=e.kind, name=e["name"])
    for k in MET:
        row[f"naive_{k}"] = e[f"{k}_A"] - e[f"{k}_U"]
        row[f"ctrl_{k}"] = (m[f"{k}_A"] - m[f"{k}_U"]).mean()
        row[f"base_{k}"] = e[f"{k}_A"] - e[f"{k}_B"]
    res.append(row)
R = pd.DataFrame(res)
print(
    "\n(2) EFFECT BEYOND BOUNCE-BACK — after (0-59 d) minus run-up (30 d before); changers vs matched slump-alike controls"
)
FOC = {
    "width:WIDER": ["whiff", "K", "bat_speed", "xwoba"],
    "width:NARROWER": ["bat_speed", "whiff", "xwoba"],
    "depth:BACK": ["chase", "sweet", "xwoba", "K"],
    "depth:UP": ["chase", "sweet", "xwoba"],
}
for kind, ms in FOC.items():
    r = R[R.kind == kind]
    print(f"   {kind} (n {len(r)})")
    for m in ms:
        nv, cv = r[f"naive_{m}"], r[f"ctrl_{m}"]
        eff = nv - cv
        se = eff.std() / np.sqrt(eff.notna().sum())
        print(
            f"      {m:9s} naive before/after {nv.mean():+6.2f} | matched non-changers' bounce-back {cv.mean():+6.2f} | "
            f"EFFECT {eff.mean():+6.2f} (SE {se:.2f}, z {eff.mean()/se:+.1f})"
        )
R.to_csv("slump_test.csv", index=False)
X.to_parquet("slump_windows.parquet")
