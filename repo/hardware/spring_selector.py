"""
spring_selector.py — 스프링을 고르고 강성 3수준을 확정한다

왜 필요한가
-----------
국내 유통되는 범용 압축스프링 세트는 강성(N/mm)을 표기하지 않는다.
선경·외경·길이만 적혀 있다. 그래서 형상에서 강성을 계산하거나
직접 재야 한다.

이 도구는 둘 다 지원한다.

  1. 형상 → 강성 계산 (주문 전, 상품 사진의 치수만으로 판단)
  2. 실측 강성 입력 (주문 후, 추를 걸어 잰 값으로 확정)

그다음 스프링 × 반경 조합을 모두 훑어 3.4절 구성표를 다시 만든다.
스프링 사양이 예상과 다르면 표가 통째로 바뀌므로,
CAD 핀 구멍 반경을 확정하기 전에 반드시 이걸 돌린다.

강성 공식
---------
    k = G · d^4 / (8 · D^3 · n_a)

    d   선경 [m]
    D   평균 코일 지름 = 외경 − 선경 [m]
    n_a 유효 감김수 (전체 감김수 − 2, 양단 연마 기준)
    G   가로탄성계수. SUS304 약 69 GPa, 피아노선(SWP) 약 79.3 GPa

계산값은 참고용이다. n_a를 눈으로 세는 데서 오차가 크게 나므로,
**부품이 도착하면 반드시 실측해서 확정한다.** 실측법은 아래 measure_note() 참고.
"""

import argparse
import itertools
import math

G_SUS304 = 69.0e9
G_SWP = 79.3e9

# 설계 상수 — 보고서 3.3절 개정판과 일치시킬 것
N_SPRINGS = 2            # 대칭 배치 개수
DEF_DEG = 8.0            # 고정 변형각
DEF_RAD = math.radians(DEF_DEG)
RES_DEG = 0.088          # AS5600 분해능
TAU_SAFE_XL430 = 1.0     # XL430 안전 토크 [N·m] (스톨 1.5의 2/3)
TAU_SAFE_XM430 = 2.7     # XM430 안전 토크 [N·m]

# 하한 제약: 스프링 토크가 관절 쿨롱 마찰보다 충분히 크지 않으면
# 링크가 아예 안 움직이거나 마찰이 거동을 지배해 강성 효과가 묻힌다.
TAU_F_ASSUMED = 0.05     # 데이터시트 기준 쿨롱 마찰 [N·m]. Phase 1-A에서 실측 후 교체
FRICTION_MARGIN = 3.0    # 마찰의 몇 배 이상이어야 하는가

# 후보 반경 [mm]. CAD 핀 구멍을 여기서 고른다.
RADII_MM = [15, 20, 25, 30, 35, 40]


def rate_from_geometry(d_mm, od_mm, turns_total, G=G_SUS304):
    """선경·외경·전체 감김수에서 강성 [N/m]을 계산한다."""
    d = d_mm / 1000.0
    D = (od_mm - d_mm) / 1000.0
    n_a = max(turns_total - 2.0, 0.5)      # 양단 연마 가정
    return G * d ** 4 / (8.0 * D ** 3 * n_a)


def measure_note():
    return """
실측법 (부품 도착 후 반드시 수행)
---------------------------------
1. 스프링을 세우고 자유장 L0를 잰다
2. 알려진 질량 m을 올린다 (동전, 너트 등을 저울로 달아둔다)
3. 눌린 길이 L을 잰다
4. k = m·g / (L0 − L)   [N/m],  단위 주의: 길이는 m

질량을 3~4개 바꿔가며 재고 직선을 그려라. 기울기가 k다.
한 점만 재면 초기 세팅 오차가 그대로 들어간다.
"""


def combos(springs, radii_mm=RADII_MM, n=N_SPRINGS):
    """스프링 × 반경 조합을 모두 계산한다."""
    out = []
    for name, k in springs.items():
        for r_mm in radii_mm:
            r = r_mm / 1000.0
            k_theta = n * k * r * r
            tau = k_theta * DEF_RAD
            x_mm = r * DEF_RAD * 1000.0
            out.append({
                "spring": name, "k": k, "r_mm": r_mm, "k_theta": k_theta,
                "tau": tau, "x_mm": x_mm,
                "xl430": tau <= TAU_SAFE_XL430,
                "xm430": tau <= TAU_SAFE_XM430,
            })
    return sorted(out, key=lambda c: c["k_theta"])


def pick_three(rows):
    """사용 가능한 조합에서 로그 등간격 3수준을 고른다.

    시뮬레이션에서 순위가 뚜렷이 뒤집히려면 최저-최고 비가 6배 이상이어야 한다.
    """
    tau_min = TAU_F_ASSUMED * FRICTION_MARGIN
    ok = [c for c in rows
          if c["xl430"] and c["x_mm"] <= 5.0 and c["tau"] >= tau_min]
    if len(ok) < 3:
        return None, ok
    lo, hi = ok[0], ok[-1]
    target = math.sqrt(lo["k_theta"] * hi["k_theta"])
    mid = min(ok, key=lambda c: abs(math.log(c["k_theta"] / target)))
    return [lo, mid, hi], ok


def report(springs):
    rows = combos(springs)
    print("=" * 78)
    print(f"스프링 × 반경 조합  (스프링 {N_SPRINGS}개, 변형각 {DEF_DEG}° 고정)")
    print("=" * 78)
    print(f"{'스프링':<12}{'강성N/mm':>10}{'반경mm':>8}{'k_theta':>10}"
          f"{'토크N·m':>10}{'압축mm':>9}{'XL430':>7}{'XM430':>7}")
    for c in rows:
        if c["k_theta"] < 0.2 or c["k_theta"] > 40:
            continue
        st = "" if c["x_mm"] <= 5.0 else "!"
        print(f"{c['spring']:<12}{c['k']/1000:>10.2f}{c['r_mm']:>8}"
              f"{c['k_theta']:>10.2f}{c['tau']:>10.2f}{c['x_mm']:>8.1f}{st:1}"
              f"{'O' if c['xl430'] else 'X':>7}{'O' if c['xm430'] else 'X':>7}")
    print("  ! = 압축량 5mm 초과, 스프링 스트로크 확인 필요")
    print(f"  전 조건 공통: 읽히는 단계 수 {DEF_DEG/RES_DEG:.0f}")

    print(f"  하한: 토크 ≥ {TAU_F_ASSUMED*FRICTION_MARGIN:.2f} N·m "
          f"(쿨롱 마찰 {TAU_F_ASSUMED} N·m의 {FRICTION_MARGIN:.0f}배)")
    three, ok = pick_three(rows)
    print("\n" + "=" * 78)
    if three is None:
        print("사용 가능한 조합이 3개 미만이다.")
        print("스프링 강성을 더 넓게 갖추거나 반경 범위를 넓혀라.")
        return
    ratio = three[2]["k_theta"] / three[0]["k_theta"]
    print(f"권장 3수준  (사용 가능 조합 {len(ok)}개, 범위 {ratio:.1f}배)")
    print("=" * 78)
    for lv, c in zip(["low", "mid", "high"], three):
        print(f"  {lv:5s}: {c['spring']:<12} 반경 {c['r_mm']}mm  "
              f"k_theta={c['k_theta']:.2f} N·m/rad  토크 {c['tau']:.2f} N·m")

    print()
    if ratio < 6.0:
        print(f"  경고: 범위가 {ratio:.1f}배로 좁다. 시뮬 기준 6배 이상이 필요하다.")
        print("  범위가 양쪽에서 눌려 있다:")
        print(f"    하한 — 스프링 토크가 쿨롱 마찰({TAU_F_ASSUMED} N·m)의"
              f" {FRICTION_MARGIN:.0f}배는 돼야 링크가 제대로 움직인다")
        print("    상한 — XL430 안전 토크 1.0 N·m")
        print("  풀 방법은 둘 중 하나다. 모터를 XM430으로 올려 상한을 넓히거나,")
        print("  Phase 1-A에서 실제 마찰을 재서 이 하한이 과했는지 확인한다.")
    else:
        print(f"  범위 {ratio:.1f}배로 충분하다. 이 3수준으로 확정한다.")

    used = sorted({c["r_mm"] for c in ok})
    print(f"\n  CAD 핀 구멍에 뚫을 반경: {used} mm")
    print("  (사용 가능한 조합에 등장하는 반경 전부. 여유분까지 뚫어두면"
          " 나중에 조건을 늘릴 수 있다.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", nargs=4, action="append", metavar=("이름", "선경mm", "외경mm", "감김수"),
                    help="형상에서 강성 계산. 여러 번 쓸 수 있다.")
    ap.add_argument("--rate", nargs=2, action="append", metavar=("이름", "N/mm"),
                    help="실측 강성 직접 입력. 여러 번 쓸 수 있다.")
    ap.add_argument("--swp", action="store_true", help="피아노선(SWP) 재질로 계산")
    a = ap.parse_args()

    springs = {}
    Gv = G_SWP if a.swp else G_SUS304
    for row in (a.geom or []):
        name, d, od, t = row[0], float(row[1]), float(row[2]), float(row[3])
        springs[name] = rate_from_geometry(d, od, t, Gv)
    for row in (a.rate or []):
        springs[row[0]] = float(row[1]) * 1000.0

    if not springs:
        # 보고서 3.4절이 전제한 값. 실제 구매품으로 반드시 교체할 것.
        print("입력이 없어 보고서 3.4절 가정값으로 돌린다.")
        print("실제 구매할 스프링 사양으로 다시 돌려라.\n")
        springs = {"연질": 300.0, "중간": 900.0, "경질": 2500.0}

    report(springs)
    print(measure_note())


if __name__ == "__main__":
    main()
