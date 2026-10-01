"""Season-level foot geometry (setup / load / stride landing) for EVERY hitter, 2024-26, from the batting-stance
page's embedded vizData. Adds landing width and width change to feet.geometry()."""

import sys, time, numpy as np, pandas as pd

sys.path.insert(0, ".")
from feet import fetch, geometry

rows = []
for yr in (2024, 2025, 2026):
    d = fetch(yr, f"{yr}-03-01", f"{yr}-11-01", f"viz/season_{yr}.html")
    for r in d:
        try:
            g = geometry(r)
        except Exception:
            continue
        side = r["side"]
        sgn = 1 if side == "L" else -1
        fr, bk = ("r", "l") if side == "L" else ("l", "r")
        mid = lambda foot, ph: np.array(
            [
                sgn * (r[f"avg_{foot}heel_x{ph}"] + r[f"avg_{foot}bigtoe_x{ph}"]) / 2,
                (r[f"avg_{foot}heel_y{ph}"] + r[f"avg_{foot}bigtoe_y{ph}"]) / 2,
            ]
        )
        g["width_land_in"] = np.linalg.norm(mid(fr, 2) - mid(bk, 2)) * 12
        g["width_change_in"] = g["width_land_in"] - np.linalg.norm(mid(fr, 0) - mid(bk, 0)) * 12
        g["back_foot_move_in"] = np.linalg.norm(mid(bk, 2) - mid(bk, 0)) * 12
        rows.append(
            dict(
                batter=r["id"],
                name=r["name"],
                side=side,
                year=str(yr),
                team=r.get("team_name"),
                contact_vs_body_in=r["avg_intercept_y_vs_batter"],
                **g,
            )
        )
    print(yr, len(d), "hitter rows", flush=True)
    time.sleep(1.2)
out = pd.DataFrame(rows)
out.to_parquet("taxonomy/geometry_all.parquet", index=False)
print(out.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
