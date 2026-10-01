"""Hitter-day outcome table (2024-26, clean zone) for the stride-dial, synthetic-control and slump analyses."""

import numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

p = pitches()
for c in ("bat_speed", "swing_length", "estimated_woba_using_speedangle", "woba_value", "launch_angle"):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
NON_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance")
ev = p.events.fillna("").astype(str)
p["pa"] = (ev != "") & ~ev.str.startswith(NON_PA) & (ev != "intent_walk")
p["xw"] = np.where(
    p.pa, np.where(p.estimated_woba_using_speedangle.notna(), p.estimated_woba_using_speedangle, p.woba_value), np.nan
)
p["K"] = p.pa & ev.str.startswith("strikeout")
p["sw"], p["wh"] = p.swing.astype(bool), p.whiff.astype(bool)
p["comp"] = p.sw & (p.bat_speed >= 50)  # competitive swings (drops check swings / bunts)
p["bip"] = p.type == "X"
p["sweet"] = p.bip & p.launch_angle.between(8, 32)
p["xwc"] = np.where(p.bip, p.estimated_woba_using_speedangle, np.nan)
p["oz_sw"] = p.out_zone & p.sw
k = ["batter", "side", "year", "game_date"]
g = p.groupby(k)
D = pd.DataFrame(
    {
        "pitches": g.size(),
        "PA": g.pa.sum(),
        "xw_sum": g.xw.sum(),
        "K": g.K.sum(),
        "sw": g.sw.sum(),
        "wh": g.wh.sum(),
        "oz": g.out_zone.sum(),
        "oz_sw": g.oz_sw.sum(),
        "bip": g.bip.sum(),
        "sweet": g.sweet.sum(),
        "xwc_sum": g.xwc.sum(),
        "xwc_n": g.xwc.count(),
        "bs_sum": p.bat_speed.where(p.comp).groupby([p[c] for c in k]).sum(),
        "bs_n": p.comp.groupby([p[c] for c in k]).sum(),
        "sl_sum": p.swing_length.where(p.comp).groupby([p[c] for c in k]).sum(),
    }
).reset_index()
D.to_parquet("hitter_day_outcomes.parquet")
print(D.shape, D.game_date.min(), D.game_date.max())
