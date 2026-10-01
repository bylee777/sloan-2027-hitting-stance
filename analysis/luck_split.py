"""Split the accounting's LUCK part (paper §4.5: run value of the ball in play minus the league value of its expected wOBA) into
DEFENSE faced, PARK, WEATHER and what is left (chance, direction of the ball, speed).
- DEFENSE: the fielding team's season Outs Above Average (Baseball Savant team leaderboard, oaa_team_<year>.csv), standardized.
- PARK: park-season means of luck estimated from VISITING hitters only (a hitter's own balls never set his home park's effect).
- WEATHER: game-window wind blowing out to centre field (mph = speed x cos, 0 indoors) and temperature (weather_games.csv: game-window
  weather from the Open-Meteo hourly archive, see the README), acting on fly balls and line drives.
Per-ball model: luck - park = b_def * OAA_z + b_wind * wind_out x {fly, liner} + b_temp * (temp - 70) x {fly, liner}, OLS clustered by batter.
Per hitter-season (150+ PA, runs per 600 PA like the accounting): defense + park + weather + other luck = luck exactly.
"""

import numpy as np, pandas as pd, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
from stance_lib import pitches

p = pitches()
for c in ("launch_speed", "launch_angle", "estimated_woba_using_speedangle", "delta_run_exp"):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
p = p.dropna(subset=["delta_run_exp"]).copy()
p["rv"] = p.delta_run_exp
p["cnt"] = p.balls.astype(str) + p.strikes.astype(str)
bip = (p.type == "X") & p.estimated_woba_using_speedangle.notna()
# xwOBA -> run value map per season x count on balls in play (as the accounting); LUCK = rv - RVx(xwOBA)
coef = {}
for (y, c), z in p[bip].groupby(["year", "cnt"]):
    A = np.column_stack([np.ones(len(z)), z.estimated_woba_using_speedangle])
    coef[(y, c)] = np.linalg.lstsq(A, z.rv.to_numpy(), rcond=None)[0]
cf = np.array([coef.get((y, c), (np.nan, np.nan)) for y, c in zip(p.year, p.cnt)])
p["LUCK"] = np.where(bip, p.rv - (cf[:, 0] + cf[:, 1] * p.estimated_woba_using_speedangle), 0.0)
NON = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "other_advance")
ev = p.events.fillna("").astype(str)
p["pa"] = (ev != "") & ~ev.str.startswith(NON)
# ---- defense faced
TEAM = {
    108: "LAA",
    109: "AZ",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "ATH",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}
O = []
for y in ("2024", "2025", "2026"):
    t = pd.read_csv(f"oaa_team_{y}.csv", encoding="utf-8-sig")
    t["year"] = y
    O.append(t[["team_id", "year", "outs_above_average"]])
O = pd.concat(O)
O["team"] = O.team_id.map(TEAM)
O["oaa_z"] = (O.outs_above_average - O.outs_above_average.mean()) / O.outs_above_average.std()
p["fteam"] = np.where(p.inning_topbot == "Top", p.home_team, p.away_team)
p["fteam"] = p.fteam.replace({"OAK": "ATH"})
p = p.merge(
    O[["team", "year", "oaa_z", "outs_above_average"]], left_on=["fteam", "year"], right_on=["team", "year"], how="left"
)
# ---- weather + park (venue) per game
# weather_games.csv keys games as season_date_HOME_AWAY (+ "_2" for a doubleheader's second game; ARI / OAK where Statcast has AZ / ATH)
Wg = pd.read_csv("weather_games.csv").rename(columns={"game_pk": "wkey"}).drop_duplicates("wkey")
Wg["wind_out"] = np.where(Wg.roof.isin(["dome", "closed", "closed_pred"]), 0.0, Wg.wind_speed_mph * Wg.wind_out_to_cf)
eng = lambda s: s.replace({"AZ": "ARI", "ATH": "OAK"})
G = p[["game_pk", "game_date", "year", "home_team", "away_team"]].drop_duplicates("game_pk").copy()
G["dh"] = G.groupby(["game_date", "home_team", "away_team"]).game_pk.rank(method="first").astype(int)
G["wkey"] = (
    G.year
    + "_"
    + G.game_date
    + "_"
    + eng(G.home_team.astype(str))
    + "_"
    + eng(G.away_team.astype(str))
    + np.where(G.dh > 1, "_2", "")
)
p = p.merge(G[["game_pk", "wkey"]], on="game_pk", how="left").merge(
    Wg[["wkey", "wind_out", "temperature_f", "venue_id"]], on="wkey", how="left"
)
print(f"games matched to weather: {G.wkey.isin(Wg.wkey).mean():.1%} of {len(G):,}")
p["park"] = p.venue_id.fillna(-1).astype(int).astype(str) + "|" + p.year
la = p.launch_angle
p["fly"] = (bip & la.between(20, 50)).astype(float)
p["liner"] = (bip & la.between(10, 20, inclusive="left")).astype(float)
p["temp_c"] = p.temperature_f - 70.0
B = p[bip & p.oaa_z.notna() & p.wind_out.notna() & p.temperature_f.notna()].copy()
print(f"balls in play: {bip.sum():,}; with defense, weather and venue: {len(B):,} ({len(B) / bip.sum():.1%})")
# joint model with park-season fixed effects: park and the HOME team's defense are separable because every team also fields on the road
for c, a, b in (
    ("w_fly", "wind_out", "fly"),
    ("w_liner", "wind_out", "liner"),
    ("t_fly", "temp_c", "fly"),
    ("t_liner", "temp_c", "liner"),
):
    B[c] = B[a] * B[b]
XV = ["oaa_z", "w_fly", "w_liner", "t_fly", "t_liner"]
Dm = B[["LUCK"] + XV].astype(float) - B.groupby("park")[["LUCK"] + XV].transform("mean").astype(float)
r = sm.OLS(Dm.LUCK, Dm[XV]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(B.batter)[0]})
print("\nper ball in play (runs; park-season fixed effects; clustered by batter):")
lab = {
    "oaa_z": "fielding team +1 SD of OAA",
    "w_fly": "fly ball, per mph blowing out",
    "w_liner": "line drive, per mph blowing out",
    "t_fly": "fly ball, per degree F warmer",
    "t_liner": "line drive, per degree F warmer",
}
for k, v in lab.items():
    print(f"   {v:34s} {r.params[k]:+.5f} (z {r.tvalues[k]:+.1f})")
bip_team = B.groupby(["fteam", "year"]).size().mean()
print(
    f"   expected if OAA converts to runs at ~0.75 per out: {-0.75 * O.outs_above_average.std() / bip_team:+.5f} per ball "
    f"(OAA SD {O.outs_above_average.std():.1f} outs over ~{bip_team:,.0f} balls in play per team-season)"
)
# park effect = visitors' mean luck net of defense and weather (a hitter's own balls never set his home park), shrunk by sample size
B["net"] = B.LUCK - (B[XV].to_numpy() @ r.params[XV].to_numpy())
vis = B[B.inning_topbot == "Top"]
pk = vis.groupby("park").net.agg(["sum", "size"])
K = 400.0
pk["eff"] = pk["sum"] / (pk["size"] + K)
B["park_eff"] = B.park.map(pk.eff).fillna(0.0)
print(
    f"   park effects (visitors, net of defense and weather): SD across park-seasons {pk.eff[pk['size'] > 1500].std():.4f} runs per ball"
)
B["DEF"] = r.params["oaa_z"] * B.oaa_z
B["WX"] = (
    r.params["w_fly"] * B.w_fly
    + r.params["w_liner"] * B.w_liner
    + r.params["t_fly"] * B.t_fly
    + r.params["t_liner"] * B.t_liner
)
B["PARK"] = B.park_eff
Bk = B.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])[
    ["game_pk", "at_bat_number", "pitch_number", "DEF", "WX", "PARK"]
]
p = p.merge(Bk, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
p[["DEF", "WX", "PARK"]] = p[["DEF", "WX", "PARK"]].fillna(0.0)
p["OTHER_LUCK"] = p.LUCK - p.DEF - p.WX - p.PARK
# ---- hitter-season
H = (
    p.groupby(["batter", "side", "year"])
    .agg(
        PA=("pa", "sum"),
        LUCK=("LUCK", "sum"),
        DEF=("DEF", "sum"),
        WX=("WX", "sum"),
        PARK=("PARK", "sum"),
        OTHER_LUCK=("OTHER_LUCK", "sum"),
    )
    .reset_index()
)
H = H[H.PA >= 150].copy()
for k in ("LUCK", "DEF", "WX", "PARK", "OTHER_LUCK"):
    H[k] = H[k] / H.PA * 600
acc = pd.read_csv("hitter_diagnosis_intended.csv", dtype={"year": str})[["batter", "side", "year", "LUCK"]].rename(
    columns={"LUCK": "LUCK_acc"}
)
H = H.merge(acc, on=["batter", "side", "year"], how="left")
print(
    f"\nhitter-seasons (150+ PA): {len(H)}; luck matches the paper's accounting: corr {H[['LUCK', 'LUCK_acc']].corr().iloc[0, 1]:.4f}, "
    f"max |diff| {(H.LUCK - H.LUCK_acc).abs().max():.3f} runs/600"
)
a = H[H.PA >= 300]
print(
    f"\nwhat luck is made of ({len(a)} hitter-seasons, 300+ PA; runs per 600 PA): SD | share of luck variance (cov/var)"
)
for k, lab2 in (
    ("DEF", "defense faced"),
    ("PARK", "park"),
    ("WX", "weather (wind + temperature)"),
    ("OTHER_LUCK", "everything else"),
):
    print(f"   {lab2:30s} SD {a[k].std():5.2f} | {100 * np.cov(a[k], a.LUCK)[0, 1] / a.LUCK.var():5.1f}%")
nx = a.assign(year=(a.year.astype(int) - 1).astype(str))
pr = a.merge(nx, on=["batter", "side", "year"], suffixes=("", "_n"))
print(
    f"\nyear-to-year r ({len(pr)} pairs): luck {pr.LUCK.corr(pr.LUCK_n):+.2f} | other luck (defense, park, weather removed) "
    f"{pr.OTHER_LUCK.corr(pr.OTHER_LUCK_n):+.2f} | park {pr.PARK.corr(pr.PARK_n):+.2f} | defense {pr.DEF.corr(pr.DEF_n):+.2f}"
)
g = H[(H.batter == 665489)].set_index("year")[["PA", "LUCK", "DEF", "PARK", "WX", "OTHER_LUCK"]]
print("\nGuerrero (runs per 600 PA):\n" + g.round(1).to_string())
ext = a.sort_values("DEF")
print(
    f"\nhardest / easiest defense faced (runs per 600 PA): {ext.DEF.min():+.2f} / {ext.DEF.max():+.2f}; park range {a.PARK.min():+.1f} / {a.PARK.max():+.1f}; "
    f"weather range {a.WX.min():+.1f} / {a.WX.max():+.1f}"
)
H.to_csv("luck_split.csv", index=False)
