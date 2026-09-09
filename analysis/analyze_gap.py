"""
analyze_gap.py — 실물 데이터에서 성분별 갭 기여도를 뽑는다

무엇을 하는가
-------------
Phase 1에서 각 성분을 실측하고, Phase 2에서 궤적 데이터를 찍고 나면
이 스크립트가 마지막 계산을 한다.

  1. 제조사 데이터시트 값만 넣은 시뮬레이션과 실물 궤적을 비교 → 기준 갭
  2. 성분을 하나씩 실측값으로 바꿔 끼우며 갭이 얼마나 줄어드는지 측정
  3. 줄어든 양이 그 성분의 기여도. 이걸로 순위를 매긴다
  4. 전부 실측값으로 바꿔도 남는 갭 = 모델 구조로 설명되지 않는 부분

핵심은 4번이다. 잔차가 크면 "파라미터를 더 정확히 재도 소용없다"는 뜻이고,
그게 이 연구가 답하려는 질문의 절반이다.

입력 파일 두 개
---------------
components.csv  — Phase 1 실측 성분값 (강성 수준별 1행)
    level, k_theta, Ia, d, tau_f, b_s, rho

trajectories.csv — Phase 2 궤적 데이터 (긴 형식)
    excitation, level, trial, t, q_cmd, q_link

xlsx도 읽는다. 확장자로 자동 판별한다.

데이터 기록 시 주의
-------------------
q_cmd는 반드시 **제어 주기와 같은 속도로** 기록한다. 제어를 1kHz로 돌리면서
로그만 200Hz로 남기면, 분석 쪽은 명령이 5ms 동안 계단처럼 유지된 줄 알고
재생한다. 그 차이가 고주파 궤적에서 구조 잔차로 둔갑한다.
데모에서 chirp/high 조건의 구조 잔차 33%가 정확히 이 현상이다.
로그를 줄여야 하면 q_link만 솎고 q_cmd는 전부 남긴다.

사용법
------
    python analyze_gap.py                      # 기본 파일명
    python analyze_gap.py --demo               # 합성 데이터로 동작 확인
    python analyze_gap.py --comp <경로> --traj <경로>
"""

import argparse
import csv
import os
import sys
import numpy as np
import mujoco

# ----------------------------------------------------------------------
# 데이터시트 값 — 아무것도 실측하지 않았을 때의 출발점
# Phase 0 전에 제조사 스펙과 CAD에서 그대로 가져온다.
# ----------------------------------------------------------------------
DATASHEET = {
    "Ia":    2.0e-4,
    "d":     0.02,
    "tau_f": 0.03,
    "b_s":   0.002,
    "rho":   1.0,
}
COMPONENTS = ["Ia", "d", "tau_f", "b_s", "rho"]

KP, KD, TAU_MAX = 12.0, 0.35, 1.5
TD = 0.004

MJCF = """
<mujoco model="sea1dof">
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="rotor" pos="0 0 0">
      <joint name="jm" type="hinge" axis="0 1 0" pos="0 0 0"
             armature="{Ia}" damping="{d}" frictionloss="{tau_f}" limited="false"/>
      <geom name="hub" type="cylinder" size="0.015 0.004" mass="1e-4"
            contype="0" conaffinity="0"/>
      <body name="link" pos="0 0 0">
        <joint name="js" type="hinge" axis="0 1 0" pos="0 0 0"
               stiffness="{k}" damping="{b_s}" springref="0" limited="false"/>
        <geom name="rod" type="capsule" fromto="0 0 0 0.15 0 0"
              size="0.006" density="{rho_rod}" contype="0" conaffinity="0"/>
        <geom name="tip" type="sphere" pos="0.15 0 0" size="0.012"
              density="{rho_tip}" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="jm" name="m" gear="1" ctrlrange="-{tmax} {tmax}"/>
  </actuator>
</mujoco>
"""


def build_model(p, k, dt):
    return mujoco.MjModel.from_xml_string(MJCF.format(
        dt=dt, Ia=p["Ia"], d=p["d"], tau_f=p["tau_f"], k=k, b_s=p["b_s"],
        rho_rod=1200.0 * p["rho"], rho_tip=2700.0 * p["rho"], tmax=TAU_MAX))


def replay(p, k, t, q_cmd):
    """실제로 준 명령을 그대로 재생해 시뮬 궤적을 만든다.

    합성 궤적이 아니라 실험에서 기록된 q_cmd를 쓰는 것이 중요하다.
    명령이 조금이라도 다르면 비교가 성립하지 않는다.

    주의: 데이터 샘플링 주기와 시뮬 적분 주기는 별개다. 200Hz로 기록했다고
    5ms로 적분하면 적분 오차가 물리 오차로 둔갑한다. 내부는 항상 1ms 이하로
    쪼개 돌리고, 기록 시점에만 값을 꺼낸다.
    """
    dt_data = float(np.median(np.diff(t)))
    sub = max(1, int(np.ceil(dt_data / 1e-3)))
    dt = dt_data / sub

    model = build_model(p, k, dt)
    data = mujoco.MjData(model)

    delay = max(1, int(round(TD / dt)))
    buf = np.zeros(delay + 1)
    out = np.empty(len(t))
    step = 0

    for i in range(len(t)):
        for _ in range(sub):
            buf[step % len(buf)] = q_cmd[i]      # 샘플 사이 명령은 유지(ZOH)
            cmd = buf[(step - delay) % len(buf)]
            tau = KP * (cmd - data.qpos[0]) - KD * data.qvel[0]
            data.ctrl[0] = np.clip(tau, -TAU_MAX, TAU_MAX)
            mujoco.mj_step(model, data)
            step += 1
        out[i] = data.qpos[0] + data.qpos[1]
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ----------------------------------------------------------------------
# 입력
# ----------------------------------------------------------------------
def load_table(path):
    """csv와 xlsx를 모두 읽어 dict 리스트로 돌려준다."""
    if path.lower().endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError:
            sys.exit("xlsx를 읽으려면 openpyxl이 필요하다: pip install openpyxl")
        ws = openpyxl.load_workbook(path, data_only=True).active
        rows = list(ws.values)
        head = [str(h).strip() for h in rows[0]]
        return [dict(zip(head, r)) for r in rows[1:] if any(v is not None for v in r)]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def num(v):
    return float(str(v).strip())


# ----------------------------------------------------------------------
# 분석
# ----------------------------------------------------------------------
def analyze(comp_rows, traj_rows, noise=0.0):
    """조건별로 성분 기여도를 계산한다."""
    comps = {r["level"]: r for r in comp_rows}

    groups = {}
    for r in traj_rows:
        key = (r["excitation"], r["level"], str(r.get("trial", "1")))
        groups.setdefault(key, []).append(r)

    per_cond = {}
    for (exc, level, trial), rows in sorted(groups.items()):
        rows.sort(key=lambda r: num(r["t"]))
        t = np.array([num(r["t"]) for r in rows])
        q_cmd = np.array([num(r["q_cmd"]) for r in rows])
        q_meas = np.array([num(r["q_link"]) for r in rows])

        if level not in comps:
            print(f"  경고: components 표에 '{level}' 수준이 없다. 건너뛴다.")
            continue
        c = comps[level]
        k = num(c["k_theta"])
        measured = {name: num(c[name]) for name in COMPONENTS}

        gap0 = rmse(replay(DATASHEET, k, t, q_cmd), q_meas)

        contrib = {}
        for name in COMPONENTS:
            p = dict(DATASHEET)
            p[name] = measured[name]
            contrib[name] = 100.0 * (gap0 - rmse(replay(p, k, t, q_cmd), q_meas)) / gap0

        gap_all = rmse(replay(measured, k, t, q_cmd), q_meas)
        residual = 100.0 * gap_all / gap0

        noise_pct = 100.0 * noise / gap0 if gap0 > 0 else 0.0
        per_cond.setdefault((exc, level), []).append(
            (gap0, contrib, residual, noise_pct))

    # 반복 시행 평균
    out = {}
    for key, trials in per_cond.items():
        n = len(trials)
        gap0 = sum(x[0] for x in trials) / n
        contrib = {c: sum(x[1][c] for x in trials) / n for c in COMPONENTS}
        residual = sum(x[2] for x in trials) / n
        noise_pct = sum(x[3] for x in trials) / n
        # 유의성 문턱: 같은 조건을 반복했을 때 기여도가 얼마나 흔들리는가.
        # 이보다 작은 기여도는 노이즈와 구분되지 않으므로 순위를 매기지 않는다.
        if n >= 3:
            spread = max(
                float(np.std([x[1][c] for x in trials], ddof=1))
                for c in COMPONENTS)
            thresh = 2.0 * spread
        else:
            thresh = noise_pct          # 반복이 부족하면 노이즈 몫으로 대체
        out[key] = (gap0, contrib, residual, noise_pct, n, thresh)
    return out


def ranked(contrib, thresh=0.0):
    """문턱을 넘는 성분만 순위를 매긴다. 나머지는 '구분 불가'로 남긴다."""
    sig = {c: v for c, v in contrib.items() if v >= thresh}
    return [c for c, _ in sorted(sig.items(), key=lambda kv: -kv[1])]


def report(results):
    excs = sorted({k[0] for k in results})
    levels = sorted({k[1] for k in results})

    print("=" * 70)
    print("성분별 갭 기여도 — 데이터시트 값에서 실측값으로 바꿨을 때 갭 감소율 [%]")
    print("=" * 70)

    for exc in excs:
        print(f"\n--- 궤적: {exc} ---")
        print(f"{'':8s}" + "".join(f"{c:>9s}" for c in COMPONENTS)
              + f"{'잔차':>9s}{'노이즈':>9s}{'구조':>8s}{'문턱':>8s}")
        for lv in levels:
            if (exc, lv) not in results:
                continue
            gap0, contrib, residual, noise_pct, n, thresh = results[(exc, lv)]
            struct = max(0.0, residual - noise_pct)
            print(f"{lv:8s}" + "".join(f"{contrib[c]:9.1f}" for c in COMPONENTS)
                  + f"{residual:9.1f}{noise_pct:9.1f}{struct:8.1f}{thresh:8.1f}")
        print("  순위:")
        for lv in levels:
            if (exc, lv) not in results:
                continue
            _, contrib, _, _, _, th = results[(exc, lv)]
            order = ranked(contrib, th)
            drop = [c for c in COMPONENTS if c not in order]
            line = ' > '.join(order) if order else "(전부 문턱 미만)"
            if drop:
                line += f"   [구분 불가: {', '.join(drop)}]"
            print(f"    {lv:6s}: {line}")

    print("\n" + "=" * 70)
    print("판정")
    print("=" * 70)
    for exc in excs:
        orders = {tuple(ranked(results[(exc, lv)][1], results[(exc, lv)][5]))
                  for lv in levels if (exc, lv) in results}
        print(f"  강성에 따른 순위 변화 [{exc:6s}]: "
              f"{'불변' if len(orders) == 1 else '바뀜'}")
    for lv in levels:
        orders = {tuple(ranked(results[(exc, lv)][1], results[(exc, lv)][5]))
                  for exc in excs if (exc, lv) in results}
        print(f"  궤적에 따른 순위 변화 [{lv:6s}]: "
              f"{'불변' if len(orders) == 1 else '바뀜'}")

    print("\n  빈 순위 칸은 기여도가 문턱 미만이라 순서를 매길 수 없다는 뜻이다.")
    worst = max(results.values(), key=lambda v: max(0.0, v[2] - v[3]))
    struct = max(0.0, worst[2] - worst[3])
    print(f"\n  최대 구조 잔차: {struct:.1f} %")
    print("  잔차 = 모든 성분을 실측값으로 바꿔도 남는 갭.")
    print("  그중 노이즈로 설명되는 몫을 뺀 나머지가 구조 잔차다.")
    print("  구조 잔차가 크면 파라미터가 아니라 모델 구조가 문제라는 뜻이며,")
    print("  그 자체가 이 연구의 결과다. 노이즈 몫은 Phase 0에서 잰")
    print("  엔코더 표준편차(--noise)로 계산하므로 반드시 넣어야 한다.")

    with open("gap_decomposition.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["excitation", "level", "component", "contribution_pct",
                    "rank", "residual_pct", "noise_pct", "structural_pct",
                    "baseline_gap_rad", "n_trials"])
        for (exc, lv), (gap0, contrib, residual, noise_pct, n, th) in sorted(results.items()):
            order = ranked(contrib, th)
            for c in COMPONENTS:
                w.writerow([exc, lv, c, f"{contrib[c]:.3f}",
                            order.index(c) + 1 if c in order else "",
                            f"{residual:.3f}", f"{noise_pct:.3f}",
                            f"{max(0.0, residual - noise_pct):.3f}",
                            f"{gap0:.6f}", n])
    print("\n  gap_decomposition.csv 저장 완료")


# ----------------------------------------------------------------------
# 데모 — 실물 데이터가 없어도 파이프라인이 도는지 확인한다
# ----------------------------------------------------------------------
def make_demo():
    """참값을 알고 있는 합성 데이터를 만든다. 코드 검증용."""
    truth = {"low": ({"Ia": 3.1e-4, "d": 0.031, "tau_f": 0.048,
                      "b_s": 0.0034, "rho": 1.12}, 0.40),
             "mid": ({"Ia": 3.1e-4, "d": 0.031, "tau_f": 0.048,
                      "b_s": 0.0034, "rho": 1.12}, 1.12),
             "high": ({"Ia": 3.1e-4, "d": 0.031, "tau_f": 0.048,
                       "b_s": 0.0034, "rho": 1.12}, 4.50)}
    rng = np.random.default_rng(0)
    comp_rows, traj_rows = [], []

    for lv, (p, k) in truth.items():
        comp_rows.append({"level": lv, "k_theta": k, **p})
        for exc in ["chirp", "slow", "stair"]:
            dt, T = 1e-3, 6.0
            t = np.arange(0, T, dt)
            if exc == "chirp":
                f0, f1 = 0.2, 12.0
                q_cmd = 0.4 * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t))
            elif exc == "slow":
                q_cmd = 0.25 * np.sin(2 * np.pi * 0.15 * t)
            else:
                q_cmd = 0.05 * np.minimum(np.floor(t / 1.0) + 1, 5)
            q_true = replay(p, k, t, q_cmd)
            for trial in (1, 2):
                q_meas = q_true + rng.normal(0, 0.0015, len(t))  # 엔코더 노이즈
                for i in range(0, len(t), 5):                    # 200Hz로 솎음
                    traj_rows.append({"excitation": exc, "level": lv, "trial": trial,
                                      "t": t[i], "q_cmd": q_cmd[i], "q_link": q_meas[i]})
    return comp_rows, traj_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="components.csv")
    ap.add_argument("--traj", default="trajectories.csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Phase 0에서 잰 엔코더 노이즈 표준편차 [rad]")
    a = ap.parse_args()

    if a.demo:
        print("합성 데이터로 파이프라인을 검증한다.\n")
        comp_rows, traj_rows = make_demo()
        if a.noise == 0.0:
            a.noise = 0.0015
    else:
        for p in (a.comp, a.traj):
            if not os.path.exists(p):
                sys.exit(f"파일이 없다: {p}\n실물 데이터가 아직이면 --demo 로 확인해라.")
        comp_rows, traj_rows = load_table(a.comp), load_table(a.traj)

    report(analyze(comp_rows, traj_rows, noise=a.noise))


if __name__ == "__main__":
    main()
