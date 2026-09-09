"""
reproduce_public.py — 공개 데이터로 갭 분석 결과를 재현한다

하드웨어 없이 돌아간다. 인터넷만 있으면 된다.

    python reproduce_public.py            # 다운로드 + 분석
    python reproduce_public.py --no-dl    # 이미 받았으면 분석만

데이터 출처
-----------
Transferable Dynamics Dataset (Max Planck Institute for Intelligent Systems)
로봇 플랫폼: Open Dynamic Robot Initiative 3-DOF finger, 1kHz
데이터: https://doi.org/10.17617/3.ZT6K7P  (Edmond / Dataverse, 공개)
코드:   https://github.com/rr-learning/transferable_dynamics_dataset

이 데이터셋을 고른 이유는 **같은 컨트롤러에 대한 실물과 시뮬 기록이
쌍으로 들어 있기 때문**이다. 갭 분석에 그대로 쓸 수 있는 형태는 흔치 않다.
"""

import argparse
import os
import urllib.request
import numpy as np

BASE = "https://edmond.mpdl.mpg.de/api/access/datafile/"
FILES = {
    "real_Sines_test_iid.npz": "194546",   # 실물, 폐루프 PD 위치제어
    "sim_Sines_test_iid.npz":  "194550",   # 시뮬, 같은 컨트롤러
    "real_GPs_test_iid.npz":   "194551",   # 실물, 개루프 토크 입력
}
DATA = "data"


def download():
    os.makedirs(DATA, exist_ok=True)
    for name, fid in FILES.items():
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            print(f"  이미 있음: {name}")
            continue
        print(f"  받는 중: {name} (약 15MB)")
        urllib.request.urlretrieve(BASE + fid, path)


def load(name):
    return np.load(os.path.join(DATA, name))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# ----------------------------------------------------------------------
def finding_1():
    """시뮬의 액추에이터가 이상적인지 확인한다.

    measured_torque와 constrained_torque의 차이를 본다. 이 둘은
    '관절에 실제로 걸린 토크'와 '걸겠다고 명령한 토크'다.
    실물이라면 마찰·지연·토크상수 오차 때문에 반드시 벌어진다.
    """
    print("\n" + "=" * 68)
    print("발견 1 — 시뮬레이터의 액추에이터가 완벽한가")
    print("=" * 68)
    for tag, name in [("시뮬", "sim_Sines_test_iid.npz"),
                      ("실물", "real_Sines_test_iid.npz")]:
        D = load(name)
        res = D["measured_torques"] - D["constrained_torques"]
        cmd = D["constrained_torques"]
        print(f"  {tag}: 잔차 RMS {rms(res):.6f} N·m   "
              f"명령 RMS {rms(cmd):.4f} N·m   "
              f"비율 {100*rms(res)/rms(cmd):5.2f}%")
    print("\n  시뮬에서 잔차가 정확히 0이면, 그 시뮬은 액추에이터 동역학을")
    print("  아예 모델링하지 않는다는 뜻이다. 성분 하나가 통째로 빠진 상태다.")


def finding_2():
    """빠진 성분이 마찰로 설명되는지 확인한다.

    잔차를 속도로 회귀해 점성 계수를 뽑고, 설명력을 본다.
    """
    print("\n" + "=" * 68)
    print("발견 2 — 그 빈자리가 마찰 모델로 메워지는가")
    print("=" * 68)
    D = load("real_Sines_test_iid.npz")
    res = (D["measured_torques"] - D["constrained_torques"]).reshape(-1, 3)
    vel = D["measured_velocities"].reshape(-1, 3)
    for j in range(3):
        v, r = vel[:, j], res[:, j]
        b = (v @ r) / (v @ v)
        r2 = 1 - np.sum((r - b * v) ** 2) / np.sum(r ** 2)
        print(f"  관절{j}: 점성계수 b={b:.5f} N·m·s/rad,  잔차 설명 {r2*100:4.1f}%")
    print("\n  설명력이 한 자릿수면 '마찰항을 넣으면 되겠지'는 거의 도움이 안 된다.")


def finding_3():
    """각도 갭이 이 데이터로 측정 가능한지 확인한다.

    t=0의 갭이 이미 크면, 그 차이는 모델 오차가 아니라
    초기 조건 오차를 포함한 값이다.
    """
    print("\n" + "=" * 68)
    print("발견 3 — 이 데이터로 각도 갭을 잴 수 있는가")
    print("=" * 68)
    R, S = load("real_Sines_test_iid.npz"), load("sim_Sines_test_iid.npz")

    dr, ds = R["desired_torques"], S["desired_torques"]
    same = rms(dr[0] - ds[0])
    cross = min(rms(dr[0] - ds[j]) for j in range(1, len(ds)))
    print(f"  롤아웃 대응 확인: 같은 인덱스 {same:.4f} vs 다른 인덱스 최소 {cross:.4f}")
    print(f"  → {'1:1로 대응한다' if same < cross else '대응 관계가 불명확하다'}")

    gap = np.abs(R["measured_angles"] - S["measured_angles"]).mean(axis=(0, 2))
    print(f"\n  t=0.00s 각도 갭: {gap[0]:.4f} rad")
    print(f"  t=14.0s 각도 갭: {gap[-1]:.4f} rad")
    print("\n  시작 시점부터 이미 벌어져 있으면 초기 조건이 안 맞춰진 것이다.")
    print("  이 차이를 '모델 오차'라고 부르면 안 된다.")


def finding_5():
    """식별되는 마찰 계수가 여기 궤적에 따라 달라지는지 확인한다.

    속도 영역이 다르면 불공정하므로 겹치는 구간으로 한정해 비교한다.
    """
    print("\n" + "=" * 68)
    print("발견 4·5 — 갭 구조가 여기 방식에 따라 달라지는가")
    print("=" * 68)
    sets = [("폐루프 Sines", "real_Sines_test_iid.npz"),
            ("개루프 GPs  ", "real_GPs_test_iid.npz")]

    for tag, name in sets:
        D = load(name)
        res = D["measured_torques"] - D["constrained_torques"]
        cmd = D["constrained_torques"]
        print(f"  {tag}: 상대 갭 {100*rms(res)/rms(cmd):5.1f}%   "
              f"속도 RMS {rms(D['measured_velocities']):.2f} rad/s")

    print(f"\n  겹치는 속도 구간(0.2~2.0 rad/s)으로 한정해 점성계수 재적합")
    fits = {}
    for tag, name in sets:
        D = load(name)
        res = (D["measured_torques"] - D["constrained_torques"]).reshape(-1, 3)
        vel = D["measured_velocities"].reshape(-1, 3)
        bs = []
        for j in range(3):
            v, r = vel[:, j], res[:, j]
            m = (np.abs(v) >= 0.2) & (np.abs(v) <= 2.0)
            bs.append((v[m] @ r[m]) / (v[m] @ v[m]))
        fits[tag] = bs
        print(f"  {tag}: " + "  ".join(f"관절{j} b={b:+.5f}" for j, b in enumerate(bs)))

    a, b = list(fits.values())
    print("\n  비율(개루프/폐루프): " + "  ".join(
        f"관절{j} {y/x:+.2f}배" for j, (x, y) in enumerate(zip(a, b))))
    print("\n  같은 관절의 같은 물리량인데 궤적에 따라 값이 달라진다.")
    print("  부호까지 뒤집히면 그 값은 마찰이 아니라, 미모델링 항이")
    print("  마찰 계수 쪽으로 흘러든 결과다. 단순 회귀로는 식별이 안 된다는 증거다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true")
    a = ap.parse_args()
    if not a.no_dl:
        print("데이터 준비")
        download()
    finding_1()
    finding_2()
    finding_3()
    finding_5()
    print("\n" + "=" * 68)
    print("자세한 해석은 findings/public-data-first-findings.md 참고")
    print("=" * 68)


if __name__ == "__main__":
    main()
