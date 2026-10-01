"""Per-batter per-DATE batting stance (depth in box etc.) from Savant's batting-stance visual.

One CSV per date in stance_daily/. Resumable. Polite 1.2 s spacing.
Columns: id,name,bat_side,side,avg_batter_y_position,avg_batter_x_position,avg_foot_sep,
avg_stance_angle,avg_intercept_y_vs_batter,avg_intercept_y_vs_plate
"""

import sys, time, subprocess
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "stance_daily"
B = "https://baseballsavant.mlb.com/visuals/batting-stance"
SEASON = {
    2024: (date(2024, 3, 20), date(2024, 9, 30)),
    2025: (date(2025, 3, 18), date(2025, 9, 28)),
    2026: (date(2026, 3, 25), date(2026, 9, 22)),
}

OUT.mkdir(exist_ok=True)
for yr in [int(a) for a in sys.argv[1:]] or list(SEASON):
    d, end = SEASON[yr]
    while d <= end:
        out = OUT / f"{d:%Y-%m-%d}.csv"
        if not out.exists():
            q = f"seasonStart={yr}&seasonEnd={yr}&dateStart={d}&dateEnd={d}" f"&minSwings=1&minGroupSwings=1&csv=true"
            body = subprocess.run(
                ["curl", "-sL", "-A", "Mozilla/5.0", f"{B}?{q}"], capture_output=True, text=True
            ).stdout
            if body.lstrip("﻿").startswith('"id"'):
                out.write_text(body)
                print(f"{d}: {body.count(chr(10))} rows", flush=True)
            else:
                print(f"{d}: NOT CSV ({body[:60]!r})", flush=True)
            time.sleep(1.2)
        d += timedelta(days=1)
print("DONE", flush=True)
