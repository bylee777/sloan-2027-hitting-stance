"""PCA's before/after change vs every 2026 regular with NO stance event, same dates (pre: season start - Jun 9;
post: Jun 10 - end). Rank = how unusual PCA's change is; also the same comparison for PCA 2025 (no narrowing)."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
D = pd.read_parquet("hitter_day_outcomes.parquet")
ev = pd.concat([pd.read_csv("width_events.csv"), pd.read_csv("events_thr4.csv")])
M = {
    "xwOBA": ("xw_sum", "PA", 1000),
    "bat speed": ("bs_sum", "bs_n", 1),
    "chase %": ("oz_sw", "oz", 100),
    "sweet-spot %": ("sweet", "bip", 100),
    "whiff %": ("wh", "sw", 100),
    "K %": ("K", "PA", 100),
}
for year, t0 in (("2026", "2026-06-10"), ("2025", "2025-06-10")):
    d = D[D.year == year].copy()
    d["post"] = d.game_date >= t0
    a = (
        d.groupby(["batter", "post"])[
            ["xw_sum", "PA", "bs_sum", "bs_n", "oz_sw", "oz", "sweet", "bip", "wh", "sw", "K"]
        ]
        .sum()
        .unstack("post")
    )
    ok = (a[("PA", False)] >= 150) & (a[("PA", True)] >= 200)
    a = a[ok]
    evy = set(ev[ev.year.astype(str) == year].batter)
    ctrl = [b for b in a.index if b not in evy and b != 691718]
    print(f"\n=== {year} (T0 {t0}); comparison group: {len(ctrl)} regulars with no stance event")
    for lab, (n, dnm, sc) in M.items():
        ch = sc * (a[(n, True)] / a[(dnm, True)] - a[(n, False)] / a[(dnm, False)])
        c = ch.loc[ctrl]
        p = ch.loc[691718]
        pct = (c < p).mean() * 100
        print(
            f"   {lab:13s} PCA {p:+7.2f} | others: mean {c.mean():+6.2f}, SD {c.std():5.2f} | PCA vs others {p - c.mean():+7.2f} "
            f"(z {(p - c.mean()) / c.std():+.1f}); PCA's change beats {pct:4.1f}% of them"
        )
