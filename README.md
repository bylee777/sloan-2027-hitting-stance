# What a Batting Stance Can and Can't Fix

Code and data for a research paper submitted to the MIT Sloan Sports Analytics Conference research paper competition (baseball).
The paper asks whether changing a batting stance changes a hitter's results. It links 2.1 million Statcast pitches from 2024 to
2026 to game-level stance geometry from Baseball Savant, compares 135 midseason stance changes with matched slumping hitters who
did not change, tests whether the answer depends on the hitter (height, body, chase tendency) or the pitcher, and uses a run-value
accounting of hitting to measure how much of a hitter's value a stance can reach.

- `analysis/`: every script behind a number in the paper, the derived data those scripts read, and each script's logged output.
- `analysis/paper/`: the paper's figure script and builder. The paper text will be added with the full paper.

## Layout

All scripts run from inside `analysis/` and read and write files next to themselves.

| Folder | Contents |
|---|---|
| `analysis/` | data pipelines, stance-change event designs, the Bayesian stance-change model and its robustness checks, the run-value accounting, within-hitter panels |
| `analysis/dl/` | the physics-based perception network (Section 4.6) |
| `analysis/taxonomy/` | the stance language and fingerprint (Sections 4.1 and 5.1) |
| `analysis/jays/`, `analysis/visual/` | figures for Sections 3 and 6 |
| `analysis/paper/` | paper figures and builder |

Each script's printed output is saved next to it as `<script>.txt`. The numbers in the paper are taken from these logs. A few
logs are named for the run rather than the script:

- `stance_bayes_v2_B.txt` / `_U.txt`: `BEFORE=B` (default) or `BEFORE=U python stance_bayes_v2.py`, the two baseline windows
  (days −90 to −31, or the 30-day run-up). `BEFORE=U` also writes `stance_bayes_v2_events_U.parquet`.
- `event_study_thr4.txt`: `python event_study.py 4` (the depth threshold in inches; 4 is the default).
- `taxonomy/archetypes_k4.txt`: `python taxonomy/archetypes.py 4` (the number of clusters; 4 is the default).
- `itt_run.txt`: written by `itt_analysis.py`.
- `dl/train_eval.txt`: `dl/train.py`, the first comparison of model architectures (`dl/model2.py` imports from it).
- `dl/robust_s<seed>_<train>to<test>[_eye<n>].txt`: written by `dl/run_robust.sh` and `dl/run_robust_eye.sh`; `s0`–`s2` is the
  random seed, `2024to2025` the training and test seasons, and `eye1`/`eye2` the eye-height variant (see the scripts).
  `dl/model2.py` also accepts tuned settings through `HP_FILE`; the tuning script is not included and no reported result
  uses it.

Some scripts also write tables and figures that the paper does not use (for example `events.csv`, `hitter_diagnosis_rel.pkl`
and `visual/v2_*.png`–`v5_*.png`); they are kept as each script's full output.

## Data

**Included**

- `stance_daily/` — Baseball Savant batting-stance values (depth in the box, distance off the plate, foot separation, stance
  angle) for every hitter and date, 2024 through September 22, 2026, collected by `crawl_stance.py`. The stance visual accepts
  date filters, so each file is a single date's snapshot.
- Derived data at the hitter-day level or coarser: hitter-day outcomes (`hitter_day_outcomes.parquet`) and power outcomes
  (`hitter_day_power.parquet`), park-adjusted stance by hitter-day (`stance_adj.parquet`), the stance-change events
  (`width_events.csv`, `events_thr4.csv`, `angle_events.csv`, `itt_events.csv`), each change's comparison with matched
  controls (`stance_bayes_v2_events_B.parquet`, `dr_counterfactual_events_{B,U}.parquet`), hitter-season foot geometry and the
  stance fingerprint (`taxonomy/`), monthly panels and model outputs.
- Small reference tables: `bio_all.csv` (MLB Stats API: height, weight, birth date), `oaa_team_<year>.csv` (Baseball Savant team
  Outs Above Average), `hand_<year>_<L|R>.csv` (Savant batting-stance values split by pitcher hand), `weather_games.csv`
  (game-time weather: hourly Open-Meteo archive at each park averaged over the game window, wind resolved toward center field
  using MLB Stats API venue bearings, zero under a closed roof; built with a separate weather pipeline that is not part of this
  repository).
- Inputs with no rebuilding script in this repository, included as-is: `front_foot_by_pitcher_hand.parquet` (front-foot
  geometry split by pitcher hand, read by `front_foot_model.py` and `foot_adjusters.py`), `visual/typical_feet.csv`,
  `visual/feet_2026.parquet` and `candidates_2026_targeted.csv` (read by `visual/figs.py`).
- Two pickles are included because figure scripts read them: `hitter_diagnosis_intended_rel.pkl` (`jays/account_figs.py`)
  and `taxonomy/_stage.pkl` (the clustering inputs, passed from `taxonomy/build.py` to `pca.py` and `archetypes.py`).

**Not included (the scripts rebuild them)**

- `raw/` — pitch-level Statcast, 2024–26 regular seasons (about 280 MB): `python crawl.py` (pybaseball; resumable; takes a few
  hours). Most analyses start from these files.
- `viz/` — cached Baseball Savant stance pages with foot coordinates, re-downloaded by `feet.py`, `taxonomy/pull_geometry.py`
  and `front_foot_monthly_pull.py`. Their parsed outputs are included.
- Pitch- and plate-appearance-level intermediates: `hitter_diagnosis_pitches.parquet` (`hitter_diagnosis.py`),
  `matchup_pa.parquet` (`matchup_model.py`), `dl/pitches_feat.parquet` and `dl/pitches_feat2.parquet` (`dl/features.py`,
  `dl/features2.py`), `vlad_bip.parquet` (`vlad_deep.py`), and posterior-draw pickles.

Statcast, Baseball Savant and MLB Stats API data are provided by MLB Advanced Media. This repository redistributes per-date stance
summaries and derived, aggregated values only; pitch-level data are downloaded from the source.

## Setup

Python 3.11. `pip install -r requirements.txt`. The perception network uses PyTorch on the CPU. The download scripts call
`curl`, which must be on the PATH.

## Reproducing

From `analysis/`:

1. **Download**: `python crawl.py` (pitch-level Statcast for 2024–26 into `raw/`; pass years, e.g. `python crawl.py 2026`, to
   fetch fewer). `stance_daily/` is included; `python crawl_stance.py` refreshes it (same year arguments).
2. **Core datasets** (each is also included, so this step can be skipped), in this order:
   - Stance changes: `events.py` (park-adjusted stance, `stance_adj.parquet`) → `event_study.py` (depth changes,
     `events_thr4.csv`) → `width.py` (width changes) → `detect_angle.py` (angle changes) → `itt_events.py` (all attempted
     changes, including reversed ones).
   - Hitter-days: `build_hitter_days.py` (`hitter_day_outcomes.parquet`) → `power_days.py`.
   - Foot geometry (downloads Savant pages into `viz/`): `feet.py`, `taxonomy/pull_geometry.py`, `front_foot_monthly_pull.py`.
   - Fingerprint: `taxonomy/build.py` → `taxonomy/pca.py` → `taxonomy/archetypes.py` → `taxonomy/reports.py`.
   - Run-value accounting: `intended_speed.py`; `hitter_diagnosis.py`, `hitter_diagnosis_intended.py`.
   - Tables that other analyses read: `front_foot_model.py` (`front_foot_panel.parquet`), `equalizer_deep.py`
     (`equalizer_events.csv`), `plate.py` (`hitter_traits.csv`), `hand_depth.py` (`hand_depth.csv`), `foot_adjusters.py`
     (`foot_adjusters.csv`, for `same_hand_success.py`) and `stance_bayes_v2.py` (`stance_bayes_v2_events_B.parquet`, for
     `dr_counterfactual.py`, `jays_predictions.py`, `loo_depth.py` and `meta_reml.py`).
3. **Analyses**, by paper section:

| Paper | Result | Script(s) | Log |
|---|---|---|---|
| §3.3 | Statcast's per-pitch zone bounds leak the swing | `zone_leak_test.py` | `zone_leak_test.txt` |
| §5.1, Fig. 2 | Stride equalizer | `equalizer_deep.py`, `stride_dial.py`, `paper/paper_figs.py` | `equalizer_deep.txt`, `stride_dial.txt` |
| §5.1 | No discrete archetypes; a stable four-dimension fingerprint | `taxonomy/archetypes.py`, `taxonomy/pca.py` | `taxonomy/archetypes_k4.txt`, `taxonomy/pca.txt` |
| §5.2, Table 2 | Contact versus power by setup lever | `power_tradeoff.py`, `power_tradeoff_angle.py` | `power_tradeoff.txt`, `power_tradeoff_angle.txt` |
| §5.2 | Intended bat speed and the contact–power trade-off | `intended_speed.py` | `intended_speed.txt` |
| §5.2 | Plate distance: reach versus jams, by season | `plate.py`, `holdout.py` | `plate.txt`, `holdout.txt` |
| §5.2 | Posture (crouch) | `crouch.py`, `posture_links.py` | `crouch.txt`, `posture_links.txt` |
| §5.3, Fig. 3 | Slumps, bounce-back and the effect beyond it | `slump_test.py` | `slump_test.txt` |
| §5.3, Tables 3–4 | Bayesian stance-change model and 2026 validation | `stance_bayes_v2.py`, `itt_analysis.py`, `loo_depth.py` | `stance_bayes_v2_B.txt`, `stance_bayes_v2_U.txt`, `itt_run.txt`, `loo_depth.txt` |
| §5.3, App. B | Posture held fixed in the stance-change models | `stance_posture.py` | `stance_posture.txt` |
| §5.3, App. B | Frequentist (REML) fit and prior scales | `meta_reml.py` | `meta_reml.txt` |
| §5.3, App. B | Learned (gradient-boosting, doubly robust) counterfactual; late-season changes | `dr_counterfactual.py`, `dr_followup.py` | `dr_counterfactual.txt`, `dr_followup.txt` |
| §5.4 | Back-and-forth tinkering | `oscillation_test.py` | `oscillation_test.txt` |
| §5.4 | Pitcher-specific setups | `matchup_model.py`, `hand_depth.py`, `same_hand_decomp.py`, `same_hand_success.py`, `pitcher_contrast.py` | matching `.txt` files |
| §4.6, §5.2, §5.4 | Perception network: pitcher-specific and hand-specific setups, eye height | `dl/features.py`, `dl/features2.py`, `dl/model2.py`, `dl/run_robust.sh`, `dl/run_robust_eye.sh`, `dl/two_position.py`, `dl/compare_eye.py` | `dl/robust_*.txt`, `dl/compare_eye.txt` |
| §5.4 | Body-specific setups | `optimal_stance_v2.py` | `optimal_stance_v2.txt` |
| §5.4 | Front foot | `front_foot_model.py` | `front_foot_model.txt` |
| §5.5, Table 5, Fig. 4 | Run-value accounting; it does not forecast better than xwOBA | `hitter_diagnosis_intended.py`, `hitter_diagnosis.py`, `hitter_diagnosis_count.py`, `next_season_test.py` | matching `.txt` files |
| §5.5 | Luck is not defense, park or weather | `luck_split.py` | `luck_split.txt` |
| §5.6 | How much a stance can reach (player plans) | `improve_plan_intended.py` | `improve_plan_intended.txt` |
| §6, Figs. 5–6 | Blue Jays accounting; Guerrero's decline | `jays/account_figs.py`, `vlad_2026.py`, `vlad_deep.py`, `jays/vlad_fig.py` | `vlad_2026.txt`, `vlad_deep.txt` |
| §6, Table 6 | Stance recommendations | `jays_predictions.py` | `jays_predictions.txt` |
| §6, Table 7 | Crow-Armstrong's 2026 in order | `pca_timeline.py`, `pca_rank.py` | `pca_timeline.txt`, `pca_rank.txt` |
| Fig. 1 | The stance language on a typical hitter | `visual/figs.py` | — |

The Bayesian model's posterior is exact (a grid over the spread of individual responses, closed form given it) and drawn with
fixed seeds; the perception-network runs record their seeds in the log names.

## Building the figures

```
cd analysis
python visual/figs.py            # Figure 1
python paper/paper_figs.py       # Figures 2 and 3
python jays/account_figs.py      # Figures 4 and 5
python jays/vlad_fig.py          # Figure 6
```

All figures are committed. The paper builder (`paper/build_paper.py`) needs the paper text, which will be added with the full
paper. Figure 3 (`paper_figs.py`) and Figure 6
(`jays/vlad_fig.py`) draw values copied into the script from `slump_test.txt` / `dr_counterfactual.txt` and `vlad_deep.txt`
rather than reading data files.
