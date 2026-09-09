"""
sweep_stiffness.py — 관절 강성을 독립변수로 둔 파라미터 민감도 스윕

무엇을 하는가
-------------
1. simlab.py의 1자유도 강체 관절을 2-body 직렬 탄성(SEA) 구조로 확장한다.
   로터와 링크를 별개 body로 두고 그 사이를 스프링으로 잇는다.
2. 강성 k를 3수준으로 두고, 각 수준에서 물리 파라미터를 하나씩 흔들어
   링크 궤적이 얼마나 변하는지 잰다.
3. 그 결과로 '성분 민감도 순위표' 3장을 만든다. 순위가 수준마다 바뀌면
   부 질문("강성이 갭 지배 성분의 순위를 바꾸는가")의 답이 시뮬 안에서 이미 나온다.
4. 덤으로 d와 tau_f의 2차원 격자를 훑어 두 파라미터가 서로 상쇄되는지
   (= 실물에서 분리 식별이 가능한지) 본다.

주의: 이 스크립트는 '실물 대비 정확도'를 재지 않는다. 시뮬 안에서
파라미터 하나가 궤적에 미치는 영향력만 잰다. 실물 데이터가 들어오면
이 순위표가 비교 기준이 된다.
"""

import csv
import numpy as np
import mujoco

# ----------------------------------------------------------------------
# 명목 파라미터 — simlab.py와 같은 값을 쓰되, SEA용 두 개를 추가했다
# ----------------------------------------------------------------------
NOMINAL = {
    "Ia":    2.0e-4,   # 로터 반사관성 [kg·m²]  (모터 회전자 관성 × 기어비²)
    "d":     0.02,     # 로터측 점성 감쇠 [N·m·s/rad]
    "tau_f": 0.03,     # 로터측 쿨롱 마찰 [N·m]
    "b_s":   0.002,    # 스프링 내부 감쇠 [N·m·s/rad]   ← SEA에서 추가
    "rho":   1.0,      # 링크 밀도 배율 (1.0 = 명목)     ← 링크 관성 대리변수
}
SWEEP_PARAMS = ["Ia", "d", "tau_f", "b_s", "rho"]

# 강성 3수준 [N·m/rad]. 낮을수록 물렁한 SEA, 높을수록 강체에 가깝다.
STIFFNESS_LEVELS = {"low": 1.0, "mid": 5.0, "high": 25.0}

# 제어기 — simlab.py와 동일. 식별 대상이 아니므로 고정한다.
KP = 12.0
KD = 0.35
TAU_MAX = 1.5

Q_BIAS = 0.0      # 엔코더 바이어스 [rad] — 스윕에서는 0 고정
TD = 0.004        # 명령 지연 [s]
T_CHIRP = 6.0     # chirp 길이 [s]
T_SLOW = 12.0     # 느린 정현파 길이 [s]
T_STAIR = 10.0    # 계단 입력 길이 [s]
PERTURB = [-0.5, -0.2, 0.2, 0.5]   # 균등 모드: 모든 파라미터를 같은 비율로

# 현실 모드: 파라미터마다 '실제로 얼마나 모르는가'가 다르다.
# 링크 질량은 CAD와 저울로 몇 % 안에 알 수 있지만, 마찰은 절반쯤 틀려도 이상하지 않다.
# 같은 ±50%를 먹이면 잘 아는 파라미터가 순위 위로 올라와 결과가 왜곡된다.
UNCERTAINTY = {
    "Ia":    0.30,   # 데이터시트 로터 관성 + 기어비 제곱 → 30% 정도
    "d":     0.50,   # 점성 감쇠, 사실상 추정치
    "tau_f": 0.50,   # 쿨롱 마찰, 온도·마모로 크게 흔들림
    "b_s":   0.50,   # 스프링 내부 감쇠, 카탈로그에 거의 없음
    "rho":   0.03,   # 링크 질량·관성, CAD + 저울로 3% 안
}


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
    """물리 파라미터 dict와 강성으로 MuJoCo 모델을 만든다."""
    xml = MJCF.format(
        dt=dt, Ia=p["Ia"], d=p["d"], tau_f=p["tau_f"],
        k=k, b_s=p["b_s"],
        rho_rod=1200.0 * p["rho"], rho_tip=2700.0 * p["rho"],
        tmax=TAU_MAX,
    )
    return mujoco.MjModel.from_xml_string(xml)


def link_inertia(model):
    """링크의 관절축 관성 [kg·m²]. 질량행렬의 js 대각항에서 읽는다."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)          # MuJoCo 3.3+
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)       # 구버전
    return M[1, 1]


def resonance_hz(k, J):
    """직렬 스프링-링크 공진 주파수 [Hz]. 여기 궤적 설계의 기준."""
    return np.sqrt(k / J) / (2.0 * np.pi)


# ----------------------------------------------------------------------
# 여기 궤적 3종
#
# 파라미터마다 '드러나는 조건'이 다르다. 하나의 궤적으로 다섯 개를 모두
# 공정하게 재는 것은 불가능하다. 그래서 성격이 다른 궤적 세 개를 쓴다.
# ----------------------------------------------------------------------

def make_chirp(f0, f1, T, amp=0.4):
    """저주파에서 고주파로 훑는다. 관성·감쇠·공진이 드러난다.

    f1은 공진 주파수 기준으로 정한다. 공진을 지나가지 못하면
    강성 차이가 데이터에 나타나지 않는다.
    """
    kr = (f1 - f0) / T
    return lambda t: amp * np.sin(2.0 * np.pi * (f0 * t + 0.5 * kr * t * t))


def make_slow(T, f=0.15, amp=0.25):
    """느린 정현파. 관성 항이 사라지고 중력과 마찰만 남는다.

    속도가 0을 지나는 순간마다 쿨롱 마찰이 방향을 뒤집으며
    궤적에 히스테리시스를 만든다. chirp에서는 이게 묻힌다.
    """
    return lambda t: amp * np.sin(2.0 * np.pi * f * t)


def make_stair(T, step=0.05, dwell=1.0, n_up=5):
    """계단 입력. 매 계단마다 정지 구간이 있다.

    정지 구간에서는 PD 제어기와 쿨롱 마찰이 균형을 이루며
    정상상태 오차 tau_f/KP를 남긴다. 마찰을 가장 직접적으로 드러내는 궤적.
    """
    def q(t):
        i = int(t / dwell)
        i = min(i, 2 * n_up - 1)
        return step * (i + 1 if i < n_up else 2 * n_up - i - 1)
    return q


EXCITATIONS = ["chirp", "slow", "stair"]


def simulate(p, k, dt, q_ref, T):
    """주어진 파라미터로 굴리고 링크 각도 시계열을 반환한다.

    제어는 로터측 PD + 지연 + 토크 포화:
        tau = sat( KP*(q_cmd(t-Td) - q_m + q_bias) - KD*qd_m )
    측정값은 링크 각도 q_l = q_m + q_s (js가 상대각이므로 더한다).
    """
    model = build_model(p, k, dt)
    data = mujoco.MjData(model)

    n = int(T / dt)
    delay_steps = int(round(TD / dt))
    cmd_buf = np.zeros(delay_steps + 1)
    out = np.empty(n)

    for i in range(n):
        t = i * dt
        cmd_buf[i % len(cmd_buf)] = q_ref(t)
        q_cmd = cmd_buf[(i - delay_steps) % len(cmd_buf)]

        q_m, qd_m = data.qpos[0], data.qvel[0]
        tau = KP * (q_cmd - q_m + Q_BIAS) - KD * qd_m
        data.ctrl[0] = np.clip(tau, -TAU_MAX, TAU_MAX)

        mujoco.mj_step(model, data)
        out[i] = data.qpos[0] + data.qpos[1]

    return out


def rel_rmse(a, b):
    """기준 궤적 대비 상대 RMSE [%]."""
    return 100.0 * np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))


def _setup(k, exc):
    """강성과 궤적 종류에 맞는 (궤적함수, dt, 길이, 공진주파수)를 정한다."""
    J = link_inertia(build_model(NOMINAL, k, 1e-3))
    f_res = resonance_hz(k, J)
    if exc == "chirp":
        f1 = 2.5 * f_res
        dt = min(1e-3, 1.0 / (40.0 * f1))
        return make_chirp(0.2, f1, T_CHIRP), dt, T_CHIRP, f_res, J
    if exc == "slow":
        return make_slow(T_SLOW), 1e-3, T_SLOW, f_res, J
    return make_stair(T_STAIR), 1e-3, T_STAIR, f_res, J


def sweep_one_level(k, exc, mode="realistic"):
    """한 (강성, 궤적) 조합에서 파라미터별 민감도를 잰다."""
    q_ref, dt, T, f_res, J = _setup(k, exc)
    base = simulate(NOMINAL, k, dt, q_ref, T)
    if not np.all(np.isfinite(base)):
        raise RuntimeError(f"k={k}, {exc}: 기준 궤적이 발산했다. dt를 줄여라.")

    out = {}
    for pname in SWEEP_PARAMS:
        if mode == "uniform":
            fracs = PERTURB
        else:
            u = UNCERTAINTY[pname]
            fracs = [-u, -0.5 * u, 0.5 * u, u]
        worst = 0.0
        for frac in fracs:
            p = dict(NOMINAL)
            p[pname] = NOMINAL[pname] * (1.0 + frac)
            worst = max(worst, rel_rmse(simulate(p, k, dt, q_ref, T), base))
        out[pname] = worst
    return out, f_res, J


def identifiability_grid(k, n=5):
    """d와 tau_f를 함께 흔들어 서로 상쇄되는지 본다.

    격자 안에서 기준 궤적과 거의 같은 궤적이 대각선 방향으로 늘어지면,
    두 파라미터가 데이터상 구분되지 않는다는 뜻이다. 실물에서도 같은 문제가 난다.
    궤적마다 따로 본다 — 어떤 궤적에서는 구분되고 어떤 궤적에서는 안 될 수 있다.
    """
    best = {}
    for exc in EXCITATIONS:
        q_ref, dt, T, _, _ = _setup(k, exc)
        base = simulate(NOMINAL, k, dt, q_ref, T)
        scales = np.linspace(0.5, 1.5, n)
        lo = (1e9, None, None)
        for sd in scales:
            for sf in scales:
                if abs(sd - 1) < 1e-9 and abs(sf - 1) < 1e-9:
                    continue
                p = dict(NOMINAL)
                p["d"] = NOMINAL["d"] * sd
                p["tau_f"] = NOMINAL["tau_f"] * sf
                e = rel_rmse(simulate(p, k, dt, q_ref, T), base)
                if e < lo[0]:
                    lo = (e, sd, sf)
        best[exc] = lo
    return best


def ranked(d):
    return [p for p, _ in sorted(d.items(), key=lambda kv: -kv[1])]


def main():
    rows = []
    print("=" * 66)
    print("파라미터 민감도 — 강성 3수준 × 여기 궤적 3종")
    print("섭동은 파라미터별 실제 불확실도. 값은 기준 궤적 대비 상대 RMSE [%]")
    print("=" * 66)

    table = {}
    for exc in EXCITATIONS:
        for lname, k in STIFFNESS_LEVELS.items():
            sens, f_res, J = sweep_one_level(k, exc)
            table[(exc, lname)] = sens
            for r, pname in enumerate(ranked(sens), 1):
                rows.append([exc, lname, k, r, pname, f"{sens[pname]:.4f}"])

    for exc in EXCITATIONS:
        print(f"\n--- 궤적: {exc} ---")
        print(f"{'':8s}" + "".join(f"{p:>9s}" for p in SWEEP_PARAMS))
        for lname in STIFFNESS_LEVELS:
            s_ = table[(exc, lname)]
            print(f"{lname:8s}" + "".join(f"{s_[p]:9.2f}" for p in SWEEP_PARAMS))
        print("  순위:")
        for lname in STIFFNESS_LEVELS:
            print(f"    {lname:5s}: {' > '.join(ranked(table[(exc, lname)]))}")

    print("\n" + "=" * 66)
    print("판정")
    print("=" * 66)
    for exc in EXCITATIONS:
        orders = {tuple(ranked(table[(exc, l)])) for l in STIFFNESS_LEVELS}
        verdict = "순위 불변" if len(orders) == 1 else "순위 바뀜"
        print(f"  강성에 따른 변화 [{exc:5s}]: {verdict}")
    for lname in STIFFNESS_LEVELS:
        orders = {tuple(ranked(table[(e, lname)])) for e in EXCITATIONS}
        verdict = "순위 불변" if len(orders) == 1 else "순위 바뀜"
        print(f"  궤적에 따른 변화 [{lname:5s}]: {verdict}")

    print("\n  tau_f(쿨롱 마찰) 민감도가 궤적별로 얼마나 달라지는가:")
    for exc in EXCITATIONS:
        vals = [table[(exc, l)]["tau_f"] for l in STIFFNESS_LEVELS]
        print(f"    {exc:6s}: {min(vals):.2f} ~ {max(vals):.2f} %")

    with open("sensitivity_ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["excitation", "level", "k", "rank", "param", "rel_rmse_pct"])
        w.writerows(rows)

    print("\n" + "=" * 66)
    print("식별 가능성 — d와 tau_f가 상쇄되는 최소 오차 (mid 강성)")
    print("=" * 66)
    best = identifiability_grid(STIFFNESS_LEVELS["mid"])
    with open("identifiability_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["excitation", "min_rel_rmse_pct", "d_scale", "tau_f_scale"])
        for exc, (e, sd, sf) in best.items():
            print(f"  {exc:6s}: {e:5.2f} %  at  d×{sd:.2f}, tau_f×{sf:.2f}")
            w.writerow([exc, f"{e:.4f}", f"{sd:.3f}", f"{sf:.3f}"])
    print("\n  이 값이 측정 노이즈보다 작은 궤적에서는 두 파라미터를 분리할 수 없다.")


if __name__ == "__main__":
    main()
