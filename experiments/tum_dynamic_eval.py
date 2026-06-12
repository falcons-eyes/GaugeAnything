"""E-dyn-0 — 핸드헬드 동적 환경 계측 스모크 (TUM RGB-D freiburg3_checkerboard_large).

질문: 손에 들고 움직이는 카메라에서 픽셀→미터 측정이 모션·블러에 따라 얼마나 열화하나.
(실측 현장 캡처의 대체 1호 — DYNAMIC_METROLOGY_DESIGN E-dyn-0)

체인: 프레임별 체커보드 코너(cv2, subpix) → Kinect depth(factor 5000)+fr3 intrinsics로
코너 역투영 → 인접 코너 3D 거리 = 사각 변 길이(미터법 측정). GT = 최저블러 상위 프레임들의
중앙값 합의. 모션 = 모캡 속도(groundtruth.txt) + 블러(Laplacian 분산).

산출: 검출률·상대오차를 속도/블러 bin별로 — "핸드헬드 열화 곡선". CC BY 4.0 (게재 클린).
실행: python experiments/tum_dynamic_eval.py --step 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEQ = Path("datasets/tum_rgbd/rgbd_dataset_freiburg3_checkerboard_large")
FX, FY, CX, CY = 535.4, 539.2, 320.1, 247.6     # freiburg3 공식 캘리브레이션
DEPTH_FACTOR = 5000.0
SIZES = [(8, 6), (7, 5), (9, 6), (6, 4), (9, 7), (5, 4), (7, 6)]


def read_list(p: Path) -> list[tuple[float, str]]:
    out = []
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            continue
        t, f = line.split()[:2]
        out.append((float(t), f))
    return out


def read_gt_speed(p: Path):
    """모캡 → (timestamps, 속도 m/s)."""
    ts, xyz = [], []
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            continue
        v = line.split()
        ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])
    ts = np.array(ts); xyz = np.array(xyz)
    dt = np.diff(ts); dx = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    spd = np.concatenate([[0], dx / np.maximum(dt, 1e-6)])
    return ts, spd


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4, help="프레임 서브샘플 간격")
    args = ap.parse_args()

    rgb = read_list(SEQ / "rgb.txt")[:: args.step]
    depth = read_list(SEQ / "depth.txt")
    d_ts = np.array([t for t, _ in depth])
    g_ts, g_spd = read_gt_speed(SEQ / "groundtruth.txt")
    print(f"=== E-dyn-0 TUM checkerboard_large: {len(rgb)} frames (step={args.step}) ===")

    size = None
    rows = []
    for t, f in rgb:
        img = cv2.imread(str(SEQ / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
        j = int(np.argmin(np.abs(g_ts - t)))
        speed = float(g_spd[j]) if abs(g_ts[j] - t) < 0.05 else None
        found = False
        if size is None:
            for sz in SIZES:
                ok, corners = cv2.findChessboardCorners(img, sz,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
                if ok:
                    size = sz; found = True; break
        else:
            found, corners = cv2.findChessboardCorners(img, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        row = {"t": t, "blur": blur, "speed": speed, "detected": bool(found)}
        if found:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            k = int(np.argmin(np.abs(d_ts - t)))
            dimg = cv2.imread(str(SEQ / depth[k][1]), cv2.IMREAD_UNCHANGED)
            if dimg is not None and abs(d_ts[k] - t) < 0.05:
                pts = corners.reshape(-1, 2)
                # 체커보드 검은 칸 = Kinect IR 흡수 → depth 홀. 3x3 중앙값 + 유효범위 필터.
                z = np.zeros(len(pts))
                Hd, Wd = dimg.shape[:2]
                for i, (px, py) in enumerate(pts):
                    x0, y0 = int(px), int(py)
                    patch = dimg[max(0, y0-1):y0+2, max(0, x0-1):x0+2].astype(float) / DEPTH_FACTOR
                    good = patch[(patch > 0.3) & (patch < 8.0)]
                    z[i] = np.median(good) if len(good) >= 3 else np.nan
                X = np.stack([(pts[:, 0] - CX) / FX * z, (pts[:, 1] - CY) / FY * z, z], 1)
                nx, ny = size
                G = X.reshape(ny, nx, 3) if len(X) == nx * ny else None
                if G is not None:
                    dists = []
                    for a, b in ((G[:, 1:], G[:, :-1]), (G[1:, :], G[:-1, :])):
                        d3 = np.linalg.norm(a - b, axis=-1).ravel()
                        dists.extend(d3[np.isfinite(d3) & (d3 > 0.005)])
                    valid = [d for d in dists if 0.02 < d < 0.3]
                    if len(valid) >= 10:
                        row["square_m"] = float(np.median(valid))
        rows.append(row)

    det = [r for r in rows if r["detected"]]
    meas = [r for r in rows if "square_m" in r]
    if not meas:
        print("측정 가능 프레임 없음 — 보드 크기 후보 확인 필요")
        return 1
    # GT 합의: 최저블러(=가장 선명) 상위 20% 프레임의 중앙값
    sharp = sorted(meas, key=lambda r: -r["blur"])[: max(5, len(meas) // 5)]
    gt = float(np.median([r["square_m"] for r in sharp]))
    for r in meas:
        r["rel_err"] = abs(r["square_m"] - gt) / gt

    # 속도/블러 bin별 열화
    def bin_stats(key, edges):
        out = {}
        for i in range(len(edges) - 1):
            sel = [r for r in meas if r.get(key) is not None and edges[i] <= r[key] < edges[i + 1]]
            n_all = [r for r in rows if r.get(key) is not None and edges[i] <= r[key] < edges[i + 1]]
            if n_all:
                out[f"{edges[i]}-{edges[i+1]}"] = {
                    "n": len(n_all), "det_rate": round(len([r for r in n_all if r['detected']]) / len(n_all), 3),
                    "rel_err_med": round(float(np.median([r["rel_err"] for r in sel])), 4) if sel else None}
        return out

    spd_stats = bin_stats("speed", [0, 0.1, 0.25, 0.5, 1.0, 3.0])
    res = {"board_size": size, "n_frames": len(rows), "det_rate": round(len(det) / len(rows), 3),
           "n_measured": len(meas), "square_consensus_m": round(gt, 5),
           "rel_err_median_all": round(float(np.median([r["rel_err"] for r in meas])), 4),
           "rel_err_p90": round(float(np.percentile([r["rel_err"] for r in meas], 90)), 4),
           "by_speed_mps": spd_stats}
    print(f"보드 {size} · 검출률 {res['det_rate']*100:.0f}% · 측정 {len(meas)}프레임")
    print(f"사각변 합의 {gt*1000:.2f}mm · 상대오차 중앙 {res['rel_err_median_all']*100:.2f}% · p90 {res['rel_err_p90']*100:.2f}%")
    print("속도 bin (m/s):", json.dumps(spd_stats, ensure_ascii=False))
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "tum_dynamic_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/tum_dynamic_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
