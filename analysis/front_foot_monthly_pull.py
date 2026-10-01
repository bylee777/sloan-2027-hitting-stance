"""Monthly foot geometry for every hitter (Savant batting-stance vizData, one window per calendar month), 2024-26."""

import sys, time, calendar, numpy as np, pandas as pd
from feet import fetch, geometry

rows = []
for yr in (2024, 2025, 2026):
    for mo in range(3, 10):
        a = f"{yr}-{mo:02d}-01"
        b = f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"
        if yr == 2026 and mo == 9:
            b = "2026-09-22"
        d = fetch(yr, a, b, f"viz/monthly/{yr}_{mo:02d}.html")
        n = 0
        for r in d:
            try:
                g = geometry(r)
            except Exception:
                continue
            fr, bk = ("r", "l") if r["side"] == "L" else ("l", "r")
            try:
                mid = lambda foot, ph: np.array(
                    [
                        (r[f"avg_{foot}heel_x{ph}"] + r[f"avg_{foot}bigtoe_x{ph}"]) / 2,
                        (r[f"avg_{foot}heel_y{ph}"] + r[f"avg_{foot}bigtoe_y{ph}"]) / 2,
                    ]
                )
                g["width_land_in"] = float(np.linalg.norm(mid(fr, 2) - mid(bk, 2)) * 12)
            except Exception:
                g["width_land_in"] = np.nan
            rows.append(dict(batter=r["id"], name=r["name"], side=r["side"], year=str(yr), month=f"{yr}-{mo:02d}", **g))
            n += 1
        print(a, "->", n, "hitters", flush=True)
        time.sleep(1.2)
pd.DataFrame(rows).to_parquet("front_foot_monthly_geometry.parquet", index=False)
print("DONE")
