# -*- coding: utf-8 -*-
"""
Jerk (Manivela Hassasiyeti) Katsayısı Duyarlılık Taraması
=========================================================
Aynı senaryolu komut profili (trim + sinus takipleri + jiter patlamaları)
farklı jerk katsayılarıyla koşulur. Dinamikler özdeştir (katsayı yalnızca
ödüle girer), bu yüzden fark tamamen ceza-ödül ilişkisini gösterir.

Metrikler:
  - total_reward : toplam bölüm ödülü (jerk cezası dahil)
  - mean_da      : adım başına ortalama |a_t - a_{t-1}| (levye sarsıntısı)
  - sign_flips   : elevator komutunun işaret değiştirme sayısı (PIO vekili)
  - stall_viol   : mach < stall_mach geçen adım sayısı
  - alt_viol     : irtifa < 1000 ft adım sayısı
  - done_count   : done düşen adım sayısı (crash)

Çıktı: analiz/jerk_sweep_results.csv + konsol tablosu
"""
import os, sys, math, random, csv, contextlib
import numpy as np

PHASE3 = r"C:\Zahit\BVR_fighterJet_AI\Phase3_Low_Level_Autopilot"
OUT_DIR = os.path.join(PHASE3, "analiz")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "jerk_sweep_results.csv")

GRID = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]   # aday jerk katsayıları
SEEDS = range(5)                            # 5 tohum (env reset randomizasyonu)
N_STEPS = 600                               # 10 saniyelik uçuş @60Hz


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


def action_profile(step, rng):
    """Sabit komut senaryosu: trim -> yavaş/hızlı sinus takibi + jiter patlamaları."""
    if step < 60:
        return np.array([0.10, 0.0, 0.30], dtype=np.float32)
    t = step - 60
    block, tt = t // 120, t % 120
    if block % 2 == 0:
        e = 0.15 * math.sin(2 * math.pi * tt / 60.0)
    else:
        e = 0.25 * math.sin(2 * math.pi * tt / 30.0)
    a = 0.10 * math.cos(2 * math.pi * tt / 47.0)
    th = 0.30 + 0.05 * math.sin(2 * math.pi * tt / 90.0)
    if tt % 40 < 3:  # 3 adımlık sert levy patlaması (bang-bang parodisi)
        e += rng.choice([-0.35, 0.35])
        a += rng.choice([-0.25, 0.25])
    return np.clip(np.array([e, a, th], dtype=np.float32), -1.0, 1.0)


def run_case(jerk_coef, seed):
    sys.path.insert(0, PHASE3)
    os.chdir(PHASE3)
    from envs.f16_env import F16Env

    rng = np.random.default_rng(seed)
    with silent():
        env = F16Env(jerk_coef=jerk_coef)
    # Env.reset() küresel np.random kullanır; aynı tohum -> aynı başlangıç
    # (böylece katsayılar arası fark yalnızca jerk cezasından gelir)
    np.random.seed(seed)
    random.seed(seed)
    env.reset()

    prev = None
    prev_e = 0.0
    total, meanda, sum_sq_da, sign_flips, stall_viol, alt_viol, done_count = 0.0, 0.0, 0.0, 0, 0, 0, 0
    for step in range(N_STEPS):
        act = action_profile(step, rng)
        s, r, done, _, _ = env.step(act)
        total += r
        if prev is not None:
            da = float(np.linalg.norm(act - prev))
            meanda += da
            sum_sq_da += da * da
            if prev_e != 0.0 and act[0] != 0.0 and (act[0] > 0) != (prev_e > 0):
                sign_flips += 1
        prev = act.copy()
        prev_e = act[0]
        if s[1] < env.stall_mach:
            stall_viol += 1
        if s[0] < 1000.0:
            alt_viol += 1
        if done:
            done_count += 1
    meanda /= N_STEPS
    return dict(jerk_coef=jerk_coef, seed=seed, total_reward=round(total, 3),
                mean_da=round(meanda, 5), sum_sq_da=round(sum_sq_da, 4),
                sign_flips=sign_flips,
                stall_viol=stall_viol, alt_viol=alt_viol, done_count=done_count)


def main():
    rows = []
    for coef in GRID:
        for seed in SEEDS:
            row = run_case(coef, seed)
            rows.append(row)
            print(f"[jerk={coef:.3f} seed={seed}] R={row['total_reward']:9.2f} "
                  f"mean_da={row['mean_da']:.4f} flips={row['sign_flips']:3d} "
                  f"stall={row['stall_viol']:3d} alt={row['alt_viol']:3d} done={row['done_count']}",
                  flush=True)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV kaydedildi: {CSV_PATH}\n")

    # Doğrusallık doğrulaması: R(k,seed) == R(0,seed) - k*sum_sq_da olmalı (dinamikler özdeş)
    base = {r["seed"]: r for r in rows if r["jerk_coef"] == 0.0}
    max_dev = 0.0
    worst = ""
    for r in rows:
        if r["jerk_coef"] == 0.0:
            continue
        pred = base[r["seed"]]["total_reward"] - r["jerk_coef"] * base[r["seed"]]["sum_sq_da"]
        dev = abs(r["total_reward"] - pred)
        if dev > max_dev:
            max_dev = dev
            worst = f"k={r['jerk_coef']}, seed={r['seed']}"
    print(f"Doğrusallık doğrulaması: max|R(k) - (R(0) - k*sum_da^2)| = {max_dev:.4f} ({worst})")
    print("  -> 0 ise ceza formülü katsayıyla birebir ölçekleniyor demektir.\n")

    print("=" * 78)
    print(f"{'jerk':>6} | {'R_ort':>10} | {'mean_da':>7} | {'flips':>5} | {'stall':>5} | {'alt':>5} | {'done':>4} | R_kayıp")
    print("-" * 78)
    base_r = np.mean([r["total_reward"] for r in rows if r["jerk_coef"] == 0.0])
    for coef in GRID:
        sub = [r for r in rows if r["jerk_coef"] == coef]
        R = np.mean([r["total_reward"] for r in sub])
        md = np.mean([r["mean_da"] for r in sub])
        fl = np.mean([r["sign_flips"] for r in sub])
        st = np.mean([r["stall_viol"] for r in sub])
        al = np.mean([r["alt_viol"] for r in sub])
        dn = np.mean([r["done_count"] for r in sub])
        kayip = (R - base_r) / abs(base_r) * 100 if base_r else 0.0
        print(f"{coef:6.2f} | {R:10.2f} | {md:7.4f} | {fl:5.1f} | {st:5.1f} | {al:5.1f} | {dn:4.1f} | %{kayip:6.2f}")
    print("=" * 78)

    R0 = base_r
    R1 = np.mean([r["total_reward"] for r in rows if r["jerk_coef"] == 0.1])
    print(f"\nYORUM: jerk=0.1 ile R={R1:.2f} (jerk=0.0 ile R={R0:.2f}; fark yalnızca ceza).")
    print("Önemli: jerk cezasının toplam ödüle etkisi k*sum_da^2 = "
          f"{0.1 * np.mean([r['sum_sq_da'] for r in rows if r['jerk_coef'] == 0.1]):.2f} puan (ödül ölçeği ~yüzler).")
    print("Öneri: mean_da/flips sabit kaldığından katsayı seçimi ödül bütçesine göre yapılır.")

if __name__ == "__main__":
    main()