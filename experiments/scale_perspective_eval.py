"""스케일 리졸버 perspective 오차 정량 (RIGOR_AUDIT F2 대응, CPU-only).

질문: 카메라 틸트에서 naive 스칼라(mm/px, 정면 가정)는 얼마나 틀리고,
homography 보정(PlaneScale)은 얼마나 회복하는가?

설정: 합성 평면(ArUco 20mm 마커 + 길이 60mm 막대), 틸트 θ로 사영 → 두 방식으로
막대 길이 측정 → GT 60mm 대비 오차%. 실 mm GT 확보 전의 기하학적 검증.

실행: python experiments/scale_perspective_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402

from gaugeanything.scale import from_aruco, make_aruco_board, plane_from_aruco  # noqa: E402

# 캐노니컬 평면: 10 px/mm. 마커 20mm=200px(여백 40px), 막대 60mm=600px.
PXMM = 10.0
MARKER_MM = 20.0
BAR_MM = 60.0
CANVAS = (1400, 800)  # (w, h)


def build_scene():
    """정면(캐노니컬) 장면 + 막대 양끝점(px)."""
    img = np.full((CANVAS[1], CANVAS[0]), 235, np.uint8)
    board = make_aruco_board(marker_size_px=int(MARKER_MM * PXMM), margin=40)
    bh, bw = board.shape
    oy, ox = 240, 120
    img[oy:oy + bh, ox:ox + bw] = board
    # 막대 (어두운 선, 마커 오른쪽 동일 평면)
    y = 400
    x1, x2 = 700, 700 + int(BAR_MM * PXMM)
    cv2.line(img, (x1, y), (x2, y), 20, thickness=9)
    return img, np.array([[x1, y], [x2, y]], np.float64)


def tilt_transform(theta_deg: float, f: float = 1600.0, Z: float = 2000.0):
    """평면을 x축 기준 θ 틸트 후 핀홀 사영하는 3×3 perspective 행렬 (캐노니컬px → 이미지px)."""
    th = np.deg2rad(theta_deg)
    cx, cy = CANVAS[0] / 2, CANVAS[1] / 2

    def proj(p):
        x, y = p[0] - cx, p[1] - cy
        y3, z3 = y * np.cos(th), Z + y * np.sin(th)
        return [f * x / z3 + cx, f * y3 / z3 + cy]

    src = np.array([[0, 0], [CANVAS[0], 0], [CANVAS[0], CANVAS[1]], [0, CANVAS[1]]], np.float32)
    dst = np.array([proj(p) for p in src], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def main():
    scene, bar = build_scene()
    rows = []
    print("=== Perspective 오차: naive mm/px vs homography (GT 막대 60mm) ===")
    print(f"{'θ(deg)':>7}{'naive 추정(mm)':>16}{'naive err%':>12}{'homog 추정(mm)':>16}{'homog err%':>12}")
    for theta in [0, 10, 20, 30, 40, 50]:
        M = tilt_transform(theta)
        warped = cv2.warpPerspective(scene, M, CANVAS, borderValue=235)
        bar_obs = cv2.perspectiveTransform(bar.reshape(1, -1, 2), M).reshape(-1, 2)

        naive = from_aruco(warped, MARKER_MM)
        plane = plane_from_aruco(warped, MARKER_MM)
        if naive is None or plane is None:
            print(f"{theta:>7}  마커 검출 실패 — 스킵")
            rows.append({"theta": theta, "detected": False})
            continue
        px_len = float(np.linalg.norm(bar_obs[0] - bar_obs[1]))
        est_naive = px_len * naive.mm_per_px
        est_homog = plane.distance_mm(bar_obs[0], bar_obs[1])
        e_n = abs(est_naive - BAR_MM) / BAR_MM * 100
        e_h = abs(est_homog - BAR_MM) / BAR_MM * 100
        rows.append({"theta": theta, "detected": True,
                     "naive_mm": round(est_naive, 2), "naive_err_pct": round(e_n, 2),
                     "homog_mm": round(est_homog, 2), "homog_err_pct": round(e_h, 2)})
        print(f"{theta:>7}{est_naive:>16.2f}{e_n:>11.1f}%{est_homog:>16.2f}{e_h:>11.1f}%")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/scale_perspective.json").write_text(
        json.dumps({"bar_mm": BAR_MM, "marker_mm": MARKER_MM, "rows": rows},
                   indent=2, ensure_ascii=False))
    print("\n결과 저장: experiments/results/scale_perspective.json")
    print("주: 막대가 마커와 떨어져 있어 naive는 '국소 스케일 차이'까지 틀림 — 현장 권고의 근거")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
