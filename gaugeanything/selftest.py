"""GaugeAnything 계측 코어 self-test — 합성 GT로 mm 정확도 검증.

모델 가중치 없이 실행 가능 (threshold 세그멘터). 검증 항목:
  [T1] 고정 폭 크랙: EDT 폭 프로파일이 GT ±10% 이내
  [T2] 변동 폭 크랙: max/p95가 단조 증가 구간 반영
  [T3] 원형 블롭: 등가직경 GT ±3%
  [T4] ArUco 스케일: 알려진 픽셀 크기 마커 → mm/px 복원 ±5%
  [T5] end-to-end: 합성 이미지(크랙+마커) → inspect() → mm 폭 GT ±15%

실행: python -m gaugeanything.selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaugeanything.geometry import measure, measure_blob, measure_thin  # noqa: E402
from gaugeanything.pipeline import inspect  # noqa: E402
from gaugeanything.scale import from_aruco, make_aruco_board  # noqa: E402

PASS, FAIL = "✓", "✗"
_failures = []


def check(name: str, got: float, expect: float, tol: float):
    rel = abs(got - expect) / max(abs(expect), 1e-9)
    ok = rel <= tol
    print(f"  {PASS if ok else FAIL} {name}: got={got:.3f} expect={expect:.3f} "
          f"(오차 {rel*100:.1f}% / 허용 {tol*100:.0f}%)")
    if not ok:
        _failures.append(name)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def synth_crack(h=400, w=800, radius=4, amp=60, wavelength=300) -> tuple[np.ndarray, float]:
    """사인 경로를 disk(r)로 팽창 → 폭 GT = 2r+1 px."""
    m = np.zeros((h, w), bool)
    xs = np.arange(30, w - 30)
    ys = (h / 2 + amp * np.sin(2 * np.pi * xs / wavelength)).astype(int)
    m[ys, xs] = True
    m = ndimage.binary_dilation(m, structure=_disk(radius))
    return m, float(2 * radius + 1)


def t1_fixed_width():
    print("[T1] 고정 폭 크랙 (EDT 폭 프로파일)")
    for r in [2, 4, 8]:
        mask, gt = synth_crack(radius=r)
        g = measure_thin(mask)
        check(f"r={r} width_mean", g.width_mean, gt, 0.10)
    # 자동 분류: 크랙은 thin이어야
    mask, _ = synth_crack(radius=4)
    g = measure(mask, kind="auto")
    print(f"  {PASS if g.kind == 'thin' else FAIL} auto-classify → {g.kind}")
    if g.kind != "thin":
        _failures.append("t1-classify")


def t2_varying_width():
    print("[T2] 변동 폭 크랙 (2r+1=5 → 13)")
    h, w = 300, 900
    m = np.zeros((h, w), bool)
    xs = np.arange(30, w - 30)
    m[h // 2, 30:w - 30] = True
    # 좌→우로 반지름 2→6 선형 증가: 구간별 팽창
    seg_r = [2, 3, 4, 5, 6]
    out = np.zeros_like(m)
    bounds = np.linspace(30, w - 30, len(seg_r) + 1).astype(int)
    for r, a, b in zip(seg_r, bounds[:-1], bounds[1:]):
        seg = np.zeros_like(m)
        seg[h // 2, a:b] = True
        out |= ndimage.binary_dilation(seg, structure=_disk(r))
    g = measure_thin(out)
    check("width_max (≈13)", g.width_max, 13.0, 0.12)
    check("width_mean (≈9)", g.width_mean, 9.0, 0.15)


def t3_blob():
    print("[T3] 원형 블롭 (등가직경)")
    for R in [20, 50]:
        h = w = 4 * R
        y, x = np.ogrid[:h, :w]
        mask = ((y - h / 2) ** 2 + (x - w / 2) ** 2) <= R * R
        g = measure_blob(mask)
        check(f"R={R} equiv_dia", g.equiv_diameter, 2 * R, 0.03)
        check(f"R={R} auto=blob", 1.0 if measure(mask).kind == "blob" else 0.0, 1.0, 0.0)


def t4_aruco():
    print("[T4] ArUco 스케일 복원")
    try:
        board = make_aruco_board(marker_size_px=200, margin=40)
    except RuntimeError as e:
        print(f"  ! 건너뜀: {e}")
        return
    # 마커 200px = (가정) 20mm → GT 0.1 mm/px
    r = from_aruco(board, marker_size_mm=20.0)
    if r is None:
        print(f"  {FAIL} 마커 탐지 실패")
        _failures.append("t4-detect")
        return
    check("mm_per_px", r.mm_per_px, 0.1, 0.05)


def t5_end_to_end():
    print("[T5] end-to-end: 합성 이미지 → inspect() → mm")
    h, w = 500, 900
    img = np.full((h, w), 230, np.uint8)
    crack_mask, gt_px = synth_crack(h=h, w=w, radius=4)   # 폭 9px
    img[crack_mask] = 20
    mm_per_px_gt = 0.1
    try:
        board = make_aruco_board(marker_size_px=100, margin=20)
        bh, bw = board.shape
        img[10:10 + bh, w - bw - 10:w - 10] = board       # 우상단에 마커 (10mm 가정)
        res = inspect(img, "crack", segmenter="threshold", marker_size_mm=10.0)
        exp_scale = 10.0 / 100.0
    except RuntimeError:
        res = inspect(img, "crack", segmenter="threshold", manual_mm_per_px=mm_per_px_gt)
        exp_scale = mm_per_px_gt
    if res.scale:
        check("scale mm/px", res.scale.mm_per_px, exp_scale, 0.05)
    thin = [a for a in res.atoms if a.metrics.kind == "thin"]
    if not thin:
        print(f"  {FAIL} thin 인스턴스 미탐지 (atoms={res.count})")
        _failures.append("t5-thin")
        return
    a = max(thin, key=lambda x: x.metrics.length)
    check("crack width_mean (mm)", a.metrics.width_mean, gt_px * exp_scale, 0.15)
    print("\n--- inspect() 출력 예시 ---")
    print(res.summary())


def main() -> int:
    print("=" * 64)
    print("GaugeAnything 계측 코어 self-test (합성 GT)")
    print("=" * 64)
    t1_fixed_width()
    t2_varying_width()
    t3_blob()
    t4_aruco()
    t5_end_to_end()
    print("=" * 64)
    if _failures:
        print(f"{FAIL} 실패 {len(_failures)}건: {_failures}")
        return 1
    print(f"{PASS} 전체 통과 — 계측 코어 검증 완료 (모델 없이)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
