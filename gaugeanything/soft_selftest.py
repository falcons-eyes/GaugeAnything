"""soft.py 수학 검증 (numpy-순수 함수). 모델/GPU 불요.

검증:
  [S1] illumination_residual: 매끄러운 평면은 잔차≈0, 국소 범프는 큰 잔차
  [S2] mura_severity: 순수 램프는 다항적합으로 제거되어 Sa≈0; 범프 추가 시 Sa↑
  [S3] soft_area: α=0.5 영역 면적 = 0.5×픽셀수 (partial volume)
  [S4] severity_score: extent 증가 → 등급 단조 상승
  [S5] area_uncertainty: α∈{0,1}이면 std=0, α=0.5에서 최대

실행: python -m gaugeanything.soft_selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaugeanything.soft import (  # noqa: E402
    area_uncertainty,
    illumination_residual,
    mura_severity,
    severity_score,
    soft_area,
)

PASS, FAIL = "✓", "✗"
_fail = []


def chk(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}  {detail}")
    if not cond:
        _fail.append(name)


def s1_residual():
    print("[S1] illumination_residual — 평면 vs 국소 범프")
    ys, xs = np.mgrid[0:128, 0:128] / 128.0
    plane = 0.5 + 0.3 * xs + 0.2 * ys                     # 매끄러운 1차 평면
    bump = np.zeros((128, 128), np.float32)
    bump[50:70, 50:70] = 0.4                              # 국소 범프
    img = (plane + bump).astype(np.float32)
    r = illumination_residual(img, order=2, mode="subtract")
    bump_resid = np.abs(r[55:65, 55:65]).mean()
    flat_resid = np.abs(r[:40, :40]).mean()
    chk("평면 잔차≈0", flat_resid < 0.02, f"flat={flat_resid:.4f}")
    chk("범프 잔차 >> 평면", bump_resid > 5 * flat_resid + 0.05, f"bump={bump_resid:.4f}")


def s2_mura():
    print("[S2] mura_severity — 순수 램프 Sa≈0, 범프 Sa↑")
    ys, xs = np.mgrid[0:128, 0:128] / 128.0
    ramp = (0.4 + 0.4 * xs).astype(np.float32)
    sev_ramp = mura_severity(ramp, order=2)
    img = ramp.copy(); img[40:80, 40:80] += 0.3
    sev_bump = mura_severity(img.astype(np.float32), order=2)
    chk("램프 Sa 작음", sev_ramp["Sa"] < 0.02, f"Sa_ramp={sev_ramp['Sa']:.4f}")
    chk("범프 Sa > 램프 Sa", sev_bump["Sa"] > sev_ramp["Sa"] * 2, f"Sa_bump={sev_bump['Sa']:.4f}")
    chk("soft_map 범위[0,1]", sev_bump["soft_map"].min() >= 0 and sev_bump["soft_map"].max() <= 1.0001)


def s3_soft_area():
    print("[S3] soft_area — partial volume")
    a = np.zeros((100, 100), np.float32); a[20:60, 20:60] = 0.5   # 40×40 영역 α=0.5
    area = soft_area(a)
    chk("면적=0.5×픽셀", abs(area - 0.5 * 40 * 40) < 1e-3, f"area={area}")
    chk("mm 스케일", abs(soft_area(a, 0.1) - 0.5 * 1600 * 0.01) < 1e-4)


def s4_severity():
    print("[S4] severity_score — extent 단조성")
    base = np.zeros((100, 100), np.float32)
    scores = []
    for cov in [0.05, 0.2, 0.5]:
        a = base.copy(); k = int(np.sqrt(cov) * 100); a[:k, :k] = 0.9
        scores.append(severity_score(a)["score"])
    chk("score 단조 증가", scores[0] < scores[1] < scores[2], f"scores={[round(s,3) for s in scores]}")
    g = severity_score((base.copy()))
    chk("빈 결함 Good", g["grade"] == "Good", f"grade={g['grade']}")


def s5_uncertainty():
    print("[S5] area_uncertainty — Bernoulli 분산")
    hard = np.zeros((50, 50), np.float32); hard[10:40, 10:40] = 1.0
    chk("α∈{0,1} → std≈0", area_uncertainty(hard) < 1e-4, f"std={area_uncertainty(hard):.4f}")
    half = np.full((50, 50), 0.5, np.float32)
    chk("α=0.5 → std>0", area_uncertainty(half) > 10, f"std={area_uncertainty(half):.2f}")


def main():
    print("=" * 60)
    print("soft.py 수학 self-test (numpy-순수)")
    print("=" * 60)
    s1_residual(); s2_mura(); s3_soft_area(); s4_severity(); s5_uncertainty()
    print("=" * 60)
    if _fail:
        print(f"{FAIL} 실패 {len(_fail)}: {_fail}"); return 1
    print(f"{PASS} 전체 통과 — soft 측정 수학 검증 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
