"""인쇄용 ArUco 보드 생성 — 실측 mm GT 수집 프로토콜용 (docs/CAPTURE_PROTOCOL.md).

A4 @ 300 DPI에 ArUco 4개(기본 30mm) + 검증용 눈금자 + 안내문.
100% 배율로 인쇄 후, 눈금자가 실제 자와 일치하는지 확인하고 사용.

실행: python experiments/make_print_board.py --out board_a4.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402

DPI = 300
MM2PX = DPI / 25.4          # 11.811 px/mm @300DPI
A4_W, A4_H = int(210 * MM2PX), int(297 * MM2PX)


def put_marker(canvas, marker_id, size_mm, top_left_mm, dictionary):
    size_px = int(round(size_mm * MM2PX))
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    x, y = int(top_left_mm[0] * MM2PX), int(top_left_mm[1] * MM2PX)
    canvas[y:y + size_px, x:x + size_px] = img
    cv2.putText(canvas, f"id={marker_id} {size_mm:.0f}mm", (x, y + size_px + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)


def draw_ruler(canvas, x_mm, y_mm, length_mm=100):
    """인쇄 배율 검증용 눈금자 (10mm 간격)."""
    x0, y0 = int(x_mm * MM2PX), int(y_mm * MM2PX)
    x1 = int((x_mm + length_mm) * MM2PX)
    cv2.line(canvas, (x0, y0), (x1, y0), 0, 3)
    for mm in range(0, length_mm + 1, 10):
        xt = int((x_mm + mm) * MM2PX)
        cv2.line(canvas, (xt, y0 - 18), (xt, y0), 0, 3)
        cv2.putText(canvas, str(mm), (xt - 12, y0 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(canvas, "VERIFY: this ruler must read true mm after printing (100% scale)",
                (x0, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="board_a4.png")
    ap.add_argument("--marker-mm", type=float, default=30.0)
    args = ap.parse_args()

    dic = cv2.aruco.DICT_4X4_50
    canvas = np.full((A4_H, A4_W), 255, np.uint8)
    cv2.putText(canvas, "GaugeAnything capture board  ·  DICT_4X4_50  ·  print at 100% / 300 DPI",
                (int(12 * MM2PX), int(12 * MM2PX)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    m = args.marker_mm
    # 4 모서리 배치 (절단해서 개별 사용 가능)
    for mid, pos in enumerate([(20, 25), (210 - 20 - m, 25), (20, 297 - 60 - m), (210 - 20 - m, 297 - 60 - m)]):
        put_marker(canvas, mid, m, pos, dic)
    draw_ruler(canvas, 55, 150, 100)
    cv2.putText(canvas, "Protocol: docs/CAPTURE_PROTOCOL.md  (place on the SAME PLANE as the defect)",
                (int(15 * MM2PX), int(285 * MM2PX)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    cv2.imwrite(args.out, canvas)
    print(f"저장: {args.out}  ({A4_W}x{A4_H}px = A4 @ {DPI}DPI, 마커 {m}mm x4 + 100mm 눈금자)")
    # self-check: 생성 보드에서 마커가 검출되고 스케일이 복원되는가
    from gaugeanything.scale import from_aruco
    r = from_aruco(canvas, m)
    if r:
        err = abs(r.mm_per_px - 1 / MM2PX) / (1 / MM2PX) * 100
        print(f"self-check: 검출 {r.n_refs}개, mm/px 복원 오차 {err:.2f}% {'✓' if err < 1 else '⚠'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
