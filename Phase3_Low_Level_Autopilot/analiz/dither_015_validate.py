# -*- coding: utf-8 -*-
"""
dither=0.15 dogrulama kosusu (3 tohum) — ana taramayi ezmez.
Referans (0.3 probing) ile karsilastirma yapar ve ayri CSV yazar.
"""
import os, sys, csv
import numpy as np

PHASE3 = r"C:\Zahit\BVR_fighterJet_AI\Phase3_Low_Level_Autopilot"
sys.path.insert(0, PHASE3)
os.chdir(PHASE3)

from analiz.dither_sweep import silent, policy, run_case
from utils.dmd_solver import RealTimeDMDc

OUT_DIR = os.path.join(PHASE3, "analiz")
CSV_PATH = os.path.join(OUT_DIR, "dither_015_validation.csv")
AMP = 0.15
SEEDS = range(3)
N_STEPS = 400
WINDOW = 300

rows = []
refs = {}
for seed in SEEDS:
    dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=WINDOW)
    A, B, stats, p_, q_, dalt_ = run_case(0.3, seed, dmd)
    refs[seed] = B
    rows.append(dict(dith_amp=0.3, seed=seed,
                     B_fro=round(float(np.linalg.norm(B)), 6),
                     B_err=0.0,
                     B_q_elev=round(float(B[6, 0]), 6),
                     B_p_ail=round(float(B[5, 1]), 6),
                     abs_p=round(p_, 5), abs_q=round(q_, 5), abs_dalt=round(dalt_, 2)))
    print(f"[referans amp=0.3 seed={seed}] B_fro={rows[-1]['B_fro']:.4f} "
          f"B_q_elev={rows[-1]['B_q_elev']} B_p_ail={rows[-1]['B_p_ail']}", flush=True)

for seed in SEEDS:
    dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=WINDOW)
    A, B, stats, p_, q_, dalt_ = run_case(AMP, seed, dmd)
    ref_norm = float(np.linalg.norm(refs[seed]))
    berr = float(np.linalg.norm(B - refs[seed]) / ref_norm) if ref_norm > 0 else 0.0
    rows.append(dict(dith_amp=AMP, seed=seed,
                     B_fro=round(float(np.linalg.norm(B)), 6),
                     B_err=round(berr, 4),
                     B_q_elev=round(float(B[6, 0]), 6),
                     B_p_ail=round(float(B[5, 1]), 6),
                     abs_p=round(p_, 5), abs_q=round(q_, 5), abs_dalt=round(dalt_, 2)))
    print(f"[dither=0.15 seed={seed}] B_fro={rows[-1]['B_fro']:.4f} "
          f"B_err={rows[-1]['B_err']} B_q_elev={rows[-1]['B_q_elev']} "
          f"B_p_ail={rows[-1]['B_p_ail']}", flush=True)

with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV kaydedildi: {CSV_PATH}\n")

def ort(k, amp):
    return np.mean([r[k] for r in rows if r["dith_amp"] == amp])

ref_q, ref_p = ort("B_q_elev", 0.3), ort("B_p_ail", 0.3)
q, p = ort("B_q_elev", AMP), ort("B_p_ail", AMP)
print("=" * 72)
print(f"{'dither':>7} | {'B_fro':>8} | {'B_err':>7} | {'B_q->elev':>9} | {'B_p->ail':>9} | {'|p|':>6} | {'|q|':>6}")
print("-" * 72)
for a in (0.15, 0.3):
    tag = "  << referans" if a == 0.3 else ""
    print(f"{a:7.2f} | {ort('B_fro', a):8.4f} | {ort('B_err', a):7.2f} | "
          f"{ort('B_q_elev', a):9.5f} | {ort('B_p_ail', a):9.5f} | "
          f"{ort('abs_p', a):6.3f} | {ort('abs_q', a):6.3f}{tag}")
print("=" * 72)

sign_q = (q > 0) == (ref_q > 0)
sign_p = (p < 0) == (ref_p < 0)
print(f"\n0.15 dogrulama: B_q_elev={q:.5f} (ref %{q/ref_q*100:.0f}, isaret {'DOGRU' if sign_q else 'YANLIS!'}), "
      f"B_p_ail={p:.5f} (ref %{p/ref_p*100:.0f}, isaret {'DOGRU' if sign_p else 'YANLIS!'})")
if sign_q and sign_p:
    print("SONUC: 0.15 DOGRULANDI - kalkan icin isaretler dogru, B_err ~1.5-2x.")
else:
    print("SONUC: 0.15 DOGRULANMADI - daha yuksek genlik gerekebilir.")