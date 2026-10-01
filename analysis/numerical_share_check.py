"""수치 오차 몫을 정말 잴 수 있는가 — 참값을 아는 상태에서 확인한다.

analyze_gap.py는 "적분 간격을 1/2로 줄였을 때의 차이 x 2"를 수치 오차 몫으로 뺀다
(SimDiff 자기 불일치, 1차 적분기의 리처드슨 추정). 그 추정이 맞는지 본다.

수치 오차만 남기는 설정:
  - 참값: 정답 파라미터로 1/8 ms 간격 시뮬 (연속 시간 대용)
  - 후보: 같은 정답 파라미터, 명령을 1 kHz로 전부 줌(ZOH 아티팩트 없음), 노이즈 없음
  → 후보와 참값의 차이는 순수하게 적분 간격 탓이다.
이 '실제 수치 오차'와 dt, dt/2, dt/4 세 번만 돌려서 낸 '추정치'를 비교한다.
참값도 1/8 ms 만큼의 적분 오차를 가지므로, 추정이 정확하면 실제/추정 = 1 − 1/8 = 0.875가 나와야 한다.

또 규칙 4("내부는 1 ms 이하")가 충분한지, 데모 조건의 기준 갭 대비 몇 %인지 본다.

python analysis/numerical_share_check.py   (약 20초)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze_gap as G  # noqa: E402

TRUTH = {"Ia": 3.1e-4, "d": 0.031, "tau_f": 0.048, "b_s": 0.0034, "rho": 1.12}
LEVELS = {"low": 0.40, "mid": 1.12, "high": 4.50}


def command(exc, t, T):
    if exc == "chirp":
        f0, f1 = 0.2, 12.0
        return 0.4 * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t))
    if exc == "slow":
        return 0.25 * np.sin(2 * np.pi * 0.15 * t)
    return 0.05 * np.minimum(np.floor(t / 1.0) + 1, 5)


def part_b():
    """명령을 전부 기록한 데모(구조 잔차 = 0이어야 함)에서 잔차를 어떻게 나눠야 맞는가.

    참 구조 잔차가 0이므로, 잔차는 노이즈 몫과 수치 오차 몫만으로 설명돼야 한다.
    그냥 더하기(노이즈 + 수치)와 제곱합(√(노이즈² + 수치²)) 중 어느 쪽이 잔차를 맞히는지 본다.
    """
    comp, traj = G.make_demo(log_every=1)
    res = G.analyze(comp, traj, noise=0.0015)
    rows = []
    for (exc, lv), v in sorted(res.items()):
        residual, noise_pct, num_pct = v[2], v[3], v[6]
        num_true = num_pct * 0.875   # 추정은 참값(1/8 ms)의 자체 오차 몫만큼 크다 (1부 참고)
        rows.append(dict(excitation=exc, level=lv, residual=residual, noise=noise_pct, numerical=num_pct,
                         linear=noise_pct + num_true, quadrature=float(np.hypot(noise_pct, num_true)),
                         structural_new=G.structural(residual, noise_pct, num_pct)))
        r = rows[-1]
        print(f"{exc:6s} {lv:5s} 잔차 {residual:5.1f}%  더하기 {r['linear']:5.1f}%  제곱합 {r['quadrature']:5.1f}%  "
              f"구조(새 방식) {r['structural_new']:4.1f}%")
    err_lin = [abs(r["linear"] - r["residual"]) for r in rows]
    err_quad = [abs(r["quadrature"] - r["residual"]) for r in rows]
    return {"n": len(rows), "median_abs_err_linear_pctpt": round(float(np.median(err_lin)), 2),
            "median_abs_err_quadrature_pctpt": round(float(np.median(err_quad)), 2),
            "max_structural_new": round(max(r["structural_new"] for r in rows), 2), "rows": rows}


def part_c():
    """노이즈가 큰 조건에서 구조 잔차를 제대로 되찾는가 — 노이즈 없는 데모를 정답으로.

    200 Hz 기록(규칙 3 위반)으로 생긴 구조 잔차를, 노이즈를 뺀 같은 데모에서 직접 잰 값과 비교한다.
    옛 방식(잔차 − 노이즈)과 새 방식(제곱합으로 빼기)을 나란히 본다.
    """
    clean = G.analyze(*G.make_demo(noise_std=0.0), noise=0.0)
    noisy = G.analyze(*G.make_demo(), noise=0.0015)
    rows = []
    for key in sorted(noisy):
        c, n = clean[key], noisy[key]
        truth = G.structural(c[2], 0.0, c[6])
        old = max(0.0, n[2] - n[3])
        new = G.structural(n[2], n[3], n[6])
        rows.append(dict(excitation=key[0], level=key[1], truth=truth, old=old, new=new))
        print(f"{key[0]:6s} {key[1]:5s} 노이즈 없는 구조 {truth:5.1f}%   옛 방식 {old:5.1f}%   새 방식 {new:5.1f}%")
    return {"n": len(rows),
            "median_abs_err_old_pctpt": round(float(np.median([abs(r["old"] - r["truth"]) for r in rows])), 2),
            "max_abs_err_old_pctpt": round(float(max(abs(r["old"] - r["truth"]) for r in rows)), 2),
            "median_abs_err_new_pctpt": round(float(np.median([abs(r["new"] - r["truth"]) for r in rows])), 2),
            "max_abs_err_new_pctpt": round(float(max(abs(r["new"] - r["truth"]) for r in rows)), 2),
            "rows": rows}


def main():
    dt, T = 1e-3, 6.0
    t = np.arange(0, T, dt)
    rows = []
    for lv, k in LEVELS.items():
        for exc in ["chirp", "slow", "stair"]:
            q_cmd = command(exc, t, T)
            truth = G.replay(TRUTH, k, t, q_cmd, refine=8)
            q1 = G.replay(TRUTH, k, t, q_cmd, refine=1)      # 1 ms
            q2 = G.replay(TRUTH, k, t, q_cmd, refine=2)      # 0.5 ms
            q4 = G.replay(TRUTH, k, t, q_cmd, refine=4)      # 0.25 ms
            actual = G.rmse(q1, truth)
            est = 2 * G.rmse(q1, q2)
            order = float(np.log2(G.rmse(q1, q2) / G.rmse(q2, q4)))
            # 데모와 같은 데이터시트 파라미터일 때의 기준 갭 (분모 비교용)
            gap0 = G.rmse(G.replay(G.DATASHEET, k, t, q_cmd), truth)
            rows.append(dict(level=lv, excitation=exc, actual_rad=actual, estimate_rad=est,
                             ratio=actual / est, observed_order=order,
                             pct_of_baseline_gap=100 * actual / gap0))
            print(f"{lv:5s} {exc:6s} 실제 {actual:.2e} rad  추정 {est:.2e} rad  비율 {actual / est:5.2f}  "
                  f"차수 {order:4.2f}  기준 갭의 {100 * actual / gap0:5.2f}%")
    r = np.array([x["ratio"] for x in rows])
    pct = np.array([x["pct_of_baseline_gap"] for x in rows])
    summary = {"n": len(rows), "ratio_median": round(float(np.median(r)), 3),
               "ratio_min": round(float(r.min()), 3), "ratio_max": round(float(r.max()), 3),
               "within_0.5_2x": int(((r >= 0.5) & (r <= 2)).sum()),
               "numerical_pct_of_gap_max": round(float(pct.max()), 2),
               "numerical_pct_of_gap_median": round(float(np.median(pct)), 2)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n2부: 명령을 전부 기록한 데모 (참 구조 잔차 = 0)")
    b = part_b()
    print(json.dumps({k: v for k, v in b.items() if k != "rows"}, ensure_ascii=False, indent=1))
    print("\n3부: 200 Hz 기록 데모의 구조 잔차 — 노이즈 없는 같은 데모가 정답")
    c = part_c()
    print(json.dumps({k: v for k, v in c.items() if k != "rows"}, ensure_ascii=False, indent=1))
    out = os.path.join(os.path.dirname(__file__), "..", "findings", "numerical_share_check.json")
    with open(out, "w") as fh:
        json.dump({"part1_estimate": {"summary": summary, "rows": rows}, "part2_decomposition": b,
                   "part3_structural_under_noise": c}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
