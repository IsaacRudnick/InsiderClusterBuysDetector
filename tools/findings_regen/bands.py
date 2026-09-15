"""Regenerate findings.py's BANDS and *_SCORE_* constants from a fresh
ensemble OOF score parquet. Definitions copied from tools/score_lab.py:
  median_excess  median(logex_21)
  win_rate       P(logex_21 > 0)
  p_loses_30pct  P(logex_21 < -0.30)      <- score_lab.decile_table.p_crash
Percentiles are of the `ens` score, low to high, over rows that have BOTH an
OOF score and a label.
"""
import os
import sys
import numpy as np, pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools import run_score_lab as lab
from tools import score_lab as sl

DATASET = sys.argv[1]
SCORES  = sys.argv[2]

df = lab.load_dataset(DATASET)
sc = pd.read_parquet(SCORES)
col = "ens" if "ens" in sc.columns else sc.columns[-1]
if len(sc) == len(df):
    df["ens"] = sc[col].to_numpy()
else:
    raise SystemExit(f"row mismatch: dataset {len(df)} vs scores {len(sc)}")

d = df[df["ens"].notna() & df[lab.GRADE_LABEL].notna()].copy()
print(f"gradeable rows with an OOF score: {len(d)}")
print(f"event_day span: {d['event_day'].min().date()} .. {d['event_day'].max().date()}")

pct = d["ens"].rank(pct=True)
BANDS = [("elevated_risk","Elevated risk",0.0,0.30),
         ("middle","Middle",0.30,0.70),
         ("top_band","Top band",0.70,0.90),
         ("above_band","Above band",0.90,1.0)]
OLD = {"elevated_risk":(-0.0268,0.436,0.0788), "middle":(-0.0083,0.461,0.0264),
       "top_band":(-0.0009,0.495,0.0121),      "above_band":(-0.0052,0.481,0.0297)}

print("\n=== BANDS ===")
print(f"{'band':14s} {'n':>6} {'median':>19} {'win':>15} {'P(<-30%)':>16}")
rows = []
for v, lbl, lo, hi in BANDS:
    m = (pct > lo) & (pct <= hi) if lo > 0 else (pct <= hi)
    g = d.loc[m, lab.GRADE_LABEL]
    med, win, crash = float(g.median()), float((g > 0).mean()), float((g < -0.30).mean())
    o = OLD[v]
    print(f"{v:14s} {len(g):>6} {med:>+9.4f} (was {o[0]:+.4f}) "
          f"{win:>6.3f} (was {o[1]:.3f}) {crash:>7.4f} (was {o[2]:.4f})")
    rows.append((v, lbl, int(lo*100), int(hi*100), round(med,4), round(win,3), round(crash,4)))

print("\n=== findings.py BANDS block ===")
for v, lbl, lo, hi, med, win, crash in rows:
    print(f'    Band("{v}", "{lbl}", {lo}, {hi}, {med}, {win}, {crash}),')

# elevated_risk crash rate by out-of-sample year
print("\n=== ELEVATED_RISK_CRASH_RATE_BY_YEAR ===")
er = d[pct <= 0.30].copy()
er["yr"] = pd.to_datetime(er["event_day"]).dt.year
by = {}
for y, g in er.groupby("yr"):
    if len(g) < 50: continue
    by[int(y)] = round(float((g[lab.GRADE_LABEL] < -0.30).mean()), 4)
print("   ", by)
if by:
    print(f"    range = ({min(by.values()):.3f}, {max(by.values()):.3f})")

# score quality audit
print("\n=== AUDIT (NEW_SCORE_* constants) ===")
a = sl.audit(d, "ens", lab.GRADE_LABEL, horizon=lab.HORIZON, name="C19 ensemble refit")
print(sl.format_audit(a))
print(f"\n  NEW_SCORE_MONTHLY_IC        = {a.ic_mean:.4f}   (was 0.0883)")
print(f"  NEW_SCORE_YEARS_POSITIVE    = ({a.years_positive}, {a.years_total})   (was (7, 7))")
print(f"  NEW_SCORE_VOL_NEUTRAL_IC    = {a.vol_neutral:.4f}   (was 0.0630)")
print(f"  NEW_SCORE_DECILE_MONOTONICITY = {a.decile_rank_corr:.2f}   (was 0.93)")
print(f"  NEW_SCORE_IC_T              = {a.ic_t:.2f}   (was 5.08)")
print(f"  NEW_SCORE_IC_CI             = ({a.ic_lo:.4f}, {a.ic_hi:.4f})   (was (0.0537, 0.1196))")
print(f"  LOW_VOL_RANKER_VOL_NEUTRAL_IC = {a.lowvol_benchmark:.4f}   (was 0.0145)")
print(f"  ic_by_year                  = {a.ic_by_year}")
print("\n  decile detail:")
print(a.deciles.to_string(index=False))
