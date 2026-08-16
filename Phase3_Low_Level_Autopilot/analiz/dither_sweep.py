# -*- coding: utf-8 -*-
"""
Dither (DMD Excitation) Genliği Duyarlılık Taraması
===================================================
Sabit düşük genlikli bir taban politikaya farklı büyüklükte rastgele gürültü
(dither) eklenir ve 400 adım (300'lük pencere dolana dek) RealTimeDMDc beslenir.
Amaç: "B matrisi canlı kalması için minimum dither" noktasını (knee) bulmak.

Metrikler:
  - B_max / B_fro : DMD B matrisinin büyüklüğü (kalkan otoritesinin vekili)
  - cond          : SVD koşul sayısı (düşük = iyi kimlik)
  - B_err         : yüksek genlikli probing (amp=0.3) referansına göre bağıl B hatası
  - |p|_ort, |q|_ort, |dAlt|_ort : dither'in uçuş yoluna kattığı gürültü
  - B[6,0], B[5,1] : q->elevator ve p->aileron otorite kazançları (°/s kalibrasyonu için)

Kalibrasyon tablosu: komut-zıplama limiti -> karşılık gelen pitch oranı değişimi
(° ve saniye cinsinden, pilotajın ~3°/s rotation kuralına ankrajlanır).

Çıktı: analiz/dither_sweep_results.csv + konsol tablosu
"""
import os, sys, math, csv, contextlib
import numpy as np

PHASE3 = r"C:\Zahit\BVR_fighterJet_AI\Phase3_Low_Level_Autopilot"
OUT_DIR = os.path.join(PHASE3, "analiz")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "dither_sweep_results.csv")

# utils.dmd_solver yalnızca numpy/scipy içerir; JSBSim 'sessiz' yüklenmez.
sys.path.insert(0, PHASE3)
from utils.dmd_solver import RealTimeDMDc

GRID = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
REF_AMP = 0.3    # "güvenilir kimlik" referansı için probing genliği
SEEDS = range(3)
N_STEPS = 400    # pencere 300; son 100 adım pencereyi yenileyip dengeye oturtur
WINDOW = 300


@contextlib.contextmanager
def silent():
    """JSBSim model yükleme gürültüsünü (stdout/stderr) susturur."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    out, err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(out, 1)
        os.dup2(err, 2)
        os.close(devnull)
        os.close(out)
        os.close(err)


def policy(t, amp, rng):
    """Düşük genlikli taban politika + dither. amp=0 -> sadece taban."""
    e = 0.06 * math.sin(2 * math.pi * t / 60.0)
    a = 0.04 * math.cos(2 * math.pi * t / 47.0)
    th = 0.30
    act = np.array([e, a, th], dtype=np.float32)
    if amp > 0.0:
        act = act + rng.uniform(-amp, amp, size=3).astype(np.float32)
    return np.clip(act, -1.0, 1.0)


def run_case(amp, seed, dmd):
    sys.path.insert(0, PHASE3)
    os.chdir(PHASE3)
    from envs.f16_env import F16Env
    from utils.dmd_solver import RealTimeDMDc

    rng = np.random.default_rng(seed + 1000)
    with silent():
        env = F16Env(jerk_coef=0.0)  # jerk kapalı: yalnızca dither etkisi ölçülsün
    state = env.reset()[0]  # gymnasium: (state, info) döner

    abs_p = abs_q = abs_dalt = 0.0
    for t in range(N_STEPS):
        act = policy(t, amp, rng)
        nxt, r, done, _, _ = env.step(act)
        dmd.add_data(state, act, nxt)
        abs_p += abs(nxt[5])
        abs_q += abs(nxt[6])
        abs_dalt += abs(nxt[0] - state[0])
        state = nxt
        if done:
            break
    abs_p /= N_STEPS
    abs_q /= N_STEPS
    abs_dalt /= N_STEPS

    A, B, stats = dmd.compute_matrices(return_stats=True)
    return A, B, stats, abs_p, abs_q, abs_dalt


def main():
    rows = []
    refs = {}  # seed -> B_ref
    for seed in SEEDS:
        dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=WINDOW)
        A, B, stats, p_, q_, dalt_ = run_case(REF_AMP, seed, dmd)
        refs[seed] = B
        rows.append(dict(dith_amp=REF_AMP, seed=seed,
                         B_max=round(float(np.abs(B).max()), 6),
                         B_fro=round(float(np.linalg.norm(B)), 6),
                         cond=round(stats["cond"], 2), rank=stats["rank"],
                         B_err=0.0,
                         B_q_elev=round(float(B[6, 0]), 6),
                         B_p_ail=round(float(B[5, 1]), 6),
                         abs_p=round(p_, 5), abs_q=round(q_, 5), abs_dalt=round(dalt_, 2)))
        print(f"[referans amp={REF_AMP} seed={seed}] B_fro={rows[-1]['B_fro']} "
              f"B_q->elev={rows[-1]['B_q_elev']} cond={rows[-1]['cond']}", flush=True)

    for amp in GRID:
        for seed in SEEDS:
            dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=WINDOW)
            A, B, stats, p_, q_, dalt_ = run_case(amp, seed, dmd)
            ref_norm = float(np.linalg.norm(refs[seed]))
            if ref_norm > 0:
                berr = float(np.linalg.norm(B - refs[seed]) / ref_norm)
            else:
                berr = 0.0
            rows.append(dict(dith_amp=amp, seed=seed,
                             B_max=round(float(np.abs(B).max()), 6),
                             B_fro=round(float(np.linalg.norm(B)), 6),
                             cond=round(stats["cond"], 2), rank=stats["rank"],
                             B_err=round(berr, 4),
                             B_q_elev=round(float(B[6, 0]), 6),
                             B_p_ail=round(float(B[5, 1]), 6),
                             abs_p=round(p_, 5), abs_q=round(q_, 5), abs_dalt=round(dalt_, 2)))
            print(f"[dither={amp:.3f} seed={seed}] B_fro={rows[-1]['B_fro']} "
                  f"B_err={rows[-1]['B_err']} cond={rows[-1]['cond']}", flush=True)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV kaydedildi: {CSV_PATH}\n")

    print("=" * 100)
    print(f"{'dither':>7} | {'B_fro':>9} | {'B_max':>9} | {'B_err':>6} | {'cond':>6} | {'|p|':>7} | {'|q|':>7} | {'|dAlt|':>7} | B_q->elev | B_p->ail")
    print("-" * 100)
    for amp in GRID + [REF_AMP]:
        sub = [r for r in rows if r["dith_amp"] == amp]
        if not sub:
            continue
        Bf = np.mean([r["B_fro"] for r in sub])
        Bm = np.mean([r["B_max"] for r in sub])
        Be = np.mean([r["B_err"] for r in sub]) if amp != REF_AMP else 0.0
        cd = np.mean([r["cond"] for r in sub])
        pp = np.mean([r["abs_p"] for r in sub])
        qq = np.mean([r["abs_q"] for r in sub])
        da = np.mean([r["abs_dalt"] for r in sub])
        bqe = np.mean([r["B_q_elev"] for r in sub])
        bpa = np.mean([r["B_p_ail"] for r in sub])
        tag = "" if amp != REF_AMP else "  << referans"
        print(f"{amp:7.3f} | {Bf:9.5f} | {Bm:9.5f} | {Be:6.3f} | {cd:6.1f} | "
              f"{pp:7.4f} | {qq:7.4f} | {da:7.2f} | {bqe:9.5f} | {bpa:9.5f}{tag}")
    print("=" * 100)

    # ---- 3°/s kalibrasyonu: komut zıplama limiti -> pitch oranı değişimi ----
    sub_knee = [r for r in rows if r["dith_amp"] == GRID[3]]  # mevcut varsayılan 0.03
    bqe = np.mean([r["B_q_elev"] for r in sub_knee])          # elevator->q kazancı (adım başına komut birimi)
    print("\nKalibrasyon (varsayılan dither 0.03'ün q->elevator kazancı kullanılarak):")
    print(f"  B[6,0] = {bqe:.6f} rad/s per komut birimi (adım başına)")
    # B, x' = A x + B u'daki adım başına kazançtır: 1 saniye (60 adım) sürdürülen
    # 1 birim levy değişimi -> q'da B*60 rad/s değişim -> deg/s cinsinden:
    deg_per_unit = bqe * 60.0 * (180.0 / math.pi)  # 1 birim komut, 1 sn sürdürülürse -> deg/s
    print(f"  1.0 tam levy hareketi (komut 0->1, 1 sn sürdürülürse) = {deg_per_unit:.1f} deg/s pitching")
    for scale in (0.25, 1.0, 4.0):
        print(f"\n  Ödül ölçeği varsayımı s={scale}:")
        for k in GRID[1:]:
            if k == 0.0:
                continue
            r_star = math.sqrt(scale / k)       # sürdürülebilir komut zıplaması (adım başına)
            r_star = min(r_star, 2.0)           # [-1,1] iki kenarı arası zıplama en fazla 2
            total_1s = min(r_star * 60.0, 2.0)  # 1 saniyede biriken komut değişimi
            deg_s = deg_per_unit * total_1s     # karşılık gelen pitch oranı değişimi
            note = "  <-- ~3 deg/s hedefe en yakın" if abs(deg_s - 3.0) < 1.5 else ""
            print(f"    jerk k={k:5.2f} -> sürdürülebilir zıplama r*={r_star:.3f}/adım "
                  f"-> 1 sn'de ~{total_1s:.2f} birim komut -> ~{deg_s:5.1f} deg/s{note}")
    print("\nNot: 3 deg/s pilotaj kurali bir manevra politikasidir; bu sistemdeki fiziksel")
    print("sinir CBF'in q bandidir (0.35 rad/s ~ 20 deg/s). Jerk katsayisi komut pürüzlülüğünü")
    print("biçimlendirir, dönüş oranini sinirlamaz.")

if __name__ == "__main__":
    main()