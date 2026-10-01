"""Paper Table 7: Pete Crow-Armstrong's 2026 in order — median game-level setup (Savant batting-stance data, not park-adjusted)
and outcomes (hitter_day_outcomes.parquet: chase on the rebuilt zone, sweet-spot share of balls in play, xwOBA per PA) by window.
"""

import glob
import pandas as pd

PCA = 691718
WINDOWS = [
    ("Opening Day - Apr 30", "2026-03-01", "2026-04-30"),
    ("May 1-17", "2026-05-01", "2026-05-17"),
    ("May 18 - Jun 9", "2026-05-18", "2026-06-09"),
    ("Jun 10 - Jul 31", "2026-06-10", "2026-07-31"),
    ("Aug 1 - Sep 22", "2026-08-01", "2026-09-22"),
]
S = []
for f in sorted(glob.glob("stance_daily/2026-*.csv")):
    s = pd.read_csv(f, encoding="utf-8-sig")
    s = s[s.id == PCA]
    if len(s):
        S.append(s.assign(game_date=f[-14:-4]))
S = pd.concat(S)
D = pd.read_parquet("hitter_day_outcomes.parquet")
D = D[(D.batter == PCA) & (D.year.astype(str) == "2026")].copy()
D["game_date"] = D.game_date.astype(str)
print(
    f"{'2026 window':22s} {'games':>5s} {'depth':>6s} {'width':>6s} {'PA':>4s} {'chase%':>7s} {'sweet%':>7s} {'xwOBA':>6s} | "
    f"out-of-zone pitches, balls in play"
)
for lab, a, b in WINDOWS:
    s = S[S.game_date.between(a, b)]
    d = D[D.game_date.between(a, b)]
    print(
        f"{lab:22s} {len(s):5d} {s.avg_batter_y_position.median():6.1f} {s.avg_foot_sep.median():6.1f} {int(d.PA.sum()):4d} "
        f"{100 * d.oz_sw.sum() / d.oz.sum():7.1f} {100 * d.sweet.sum() / d.bip.sum():7.1f} {d.xw_sum.sum() / d.PA.sum():6.3f} | "
        f"{int(d.oz.sum())}, {int(d.bip.sum())}"
    )
