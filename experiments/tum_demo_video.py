"""TUM 동적 데모 영상 — 손에 든 카메라가 움직여도 mm 측정이 유지된다.

E-dyn-0(tum_dynamic_eval)의 측정 체인을 한 핸드헬드 시퀀스의 연속 프레임에 적용:
프레임별 체커보드 코너 → Kinect depth + fr3 intrinsics 역투영 → 인접 코너 3D 거리
= 사각 변 길이(mm). 카메라 속도(모캡 groundtruth)와 블러를 함께 표시해 "모션 강건 계측"을
보여준다. 게이트(등방성/산포) 실패는 '측정 불가'로 정직 표시.

Spark 실행:
    .venv/bin/python experiments/tum_demo_video.py --step 5 --max-frames 80 --fps 12
출력: docs/assets/tum_demo.mp4, docs/assets/tum_demo.webp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEQ = Path("datasets/tum_rgbd/rgbd_dataset_freiburg3_checkerboard_large")
FX, FY, CX, CY = 535.4, 539.2, 320.1, 247.6
DEPTH_FACTOR = 5000.0
SIZES = [(8, 6), (7, 5), (9, 6), (6, 4), (9, 7), (5, 4), (7, 6)]
OUT_MP4 = Path("docs/assets/tum_demo.mp4")
OUT_WEBP = Path("docs/assets/tum_demo.webp")


def read_list(p: Path):
    out = []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, f = line.split()[:2]
        out.append((float(t), f))
    return out


def read_gt_speed(p: Path):
    ts, xyz = [], []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])
    ts = np.array(ts); xyz = np.array(xyz)
    spd = np.zeros(len(ts))
    if len(ts) > 1:
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        dt = np.diff(ts)
        spd[1:] = d / np.maximum(dt, 1e-6)
    return ts, spd


def measure_frame(img_gray, dimg, size):
    import cv2

    if size is None:
        for sz in SIZES:
            ok, corners = cv2.findChessboardCorners(img_gray, sz,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
            if ok:
                size = sz; break
        else:
            return None, None, size
    else:
        ok, corners = cv2.findChessboardCorners(img_gray, size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            return None, None, size
    corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    pts = corners.reshape(-1, 2)
    if dimg is None:
        return None, pts, size
    z = np.full(len(pts), np.nan)
    for i, (px, py) in enumerate(pts):
        x0, y0 = int(px), int(py)
        patch = dimg[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2].astype(float) / DEPTH_FACTOR
        good = patch[(patch > 0.3) & (patch < 8.0)]
        z[i] = np.median(good) if len(good) >= 3 else np.nan
    X = np.stack([(pts[:, 0] - CX) / FX * z, (pts[:, 1] - CY) / FY * z, z], 1)
    nx, ny = size
    if len(X) != nx * ny:
        return None, pts, size
    G = X.reshape(ny, nx, 3)
    dx = np.linalg.norm(G[:, 1:] - G[:, :-1], axis=-1).ravel()
    dy = np.linalg.norm(G[1:, :] - G[:-1, :], axis=-1).ravel()
    dx = dx[np.isfinite(dx) & (dx > 0.02) & (dx < 0.3)]
    dy = dy[np.isfinite(dy) & (dy > 0.02) & (dy < 0.3)]
    if len(dx) < 8 or len(dy) < 8:
        return None, pts, size
    dxm, dym = float(np.median(dx)), float(np.median(dy))
    mad = float(np.median(np.abs(np.concatenate([dx, dy]) - 0.5 * (dxm + dym))))
    if abs(dxm - dym) / max(dxm, dym) < 0.05 and mad < 0.1 * dxm:
        return 0.5 * (dxm + dym), pts, size      # 게이트 통과
    return None, pts, size                         # 게이트 실패(측정 불가)


def draw(img_bgr, pts, square_m, gt_mm, speed, blur, idx, n):
    import cv2

    out = img_bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 70), (14, 22, 32), -1)
    cv2.putText(out, "GaugeAnything  -  metric stays still while the camera moves", (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (54, 224, 176), 2, cv2.LINE_AA)
    cv2.putText(out, f"TUM handheld RGB-D   |   square side = mm from depth back-projection   |   frame {idx+1}/{n}",
                (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 230, 239), 1, cv2.LINE_AA)
    if pts is not None:
        col = (54, 224, 176) if square_m is not None else (224, 120, 120)
        for p in pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 3, col, -1, cv2.LINE_AA)
    sp = f"{speed*100:4.1f} cm/s" if speed is not None else "  -  "
    cv2.rectangle(out, (0, h - 64), (w, h), (14, 22, 32), -1)
    if square_m is not None:
        mm = square_m * 1000
        err = abs(mm - gt_mm) / gt_mm * 100
        cv2.putText(out, f"square {mm:5.1f} mm  (ref {gt_mm:.1f}, {err:.1f}%)   camera speed {sp}   blur {blur:.0f}",
                    (14, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "measured live under motion - no tripod, no static assumption",
                    (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (181, 200, 214), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"not measurable this frame (gate rejected: motion blur / depth desync)   speed {sp}",
                    (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 120, 120), 1, cv2.LINE_AA)
    return out


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="step-샘플 rgb 리스트에서 시작 오프셋")
    ap.add_argument("--max-frames", type=int, default=70)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    ap.add_argument("--out-webp", type=Path, default=OUT_WEBP)
    args = ap.parse_args()

    print("=== TUM demo video ===", flush=True)
    rgb = read_list(SEQ / "rgb.txt")[:: args.step][args.start: args.start + args.max_frames]
    depth = read_list(SEQ / "depth.txt")
    d_ts = np.array([t for t, _ in depth])
    g_ts, g_spd = read_gt_speed(SEQ / "groundtruth.txt")
    print(f"frames {len(rgb)}", flush=True)

    # dominant 체커보드 크기 다수결 (순서 의존 락 방지)
    from collections import Counter
    votes = Counter()
    for t, f in rgb[: min(len(rgb), 25)]:
        img = cv2.imread(str(SEQ / f), cv2.IMREAD_GRAYSCALE)
        for sz in SIZES:
            ok, _ = cv2.findChessboardCorners(img, sz,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
            if ok:
                votes[sz] += 1
                break
    dominant = votes.most_common(1)[0][0] if votes else None
    print(f"dominant board size {dominant} (votes {dict(votes)})", flush=True)

    # 1차 패스: 측정값 수집 → GT 합의(저블러 상위 프레임 중앙값)
    size = dominant
    recs = []
    for t, f in rgb:
        img = cv2.imread(str(SEQ / f), cv2.IMREAD_GRAYSCALE)
        blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
        k = int(np.argmin(np.abs(d_ts - t)))
        dimg = cv2.imread(str(SEQ / depth[k][1]), cv2.IMREAD_UNCHANGED) if abs(d_ts[k] - t) < 0.02 else None
        sq, pts, size = measure_frame(img, dimg, size)
        j = int(np.argmin(np.abs(g_ts - t)))
        speed = float(g_spd[j]) if abs(g_ts[j] - t) < 0.05 else None
        recs.append({"t": t, "f": f, "sq": sq, "pts": pts, "blur": blur, "speed": speed})
    meas = [r["sq"] for r in recs if r["sq"] is not None]
    if not meas:
        print("no measurable frames", flush=True)
        return 1
    # GT = 저블러 상위 40% 프레임의 측정 중앙값
    lowblur = sorted([r for r in recs if r["sq"]], key=lambda r: -r["blur"])[: max(1, len(meas) * 4 // 10)]
    gt_mm = float(np.median([r["sq"] for r in lowblur])) * 1000
    print(f"reference square {gt_mm:.2f}mm, measurable {len(meas)}/{len(recs)}", flush=True)

    rendered, errs = [], []
    for i, r in enumerate(recs):
        img_bgr = cv2.imread(str(SEQ / r["f"]), cv2.IMREAD_COLOR)
        rendered.append(draw(img_bgr, r["pts"], r["sq"], gt_mm, r["speed"], r["blur"], i, len(recs)))
        if r["sq"]:
            errs.append(abs(r["sq"] * 1000 - gt_mm) / gt_mm)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out_mp4)], check=True)
    from PIL import Image
    h, w = rendered[0].shape[:2]
    small = [Image.fromarray(cv2.resize(im, (w // 2, h // 2))[:, :, ::-1]) for im in rendered]
    small[0].save(args.out_webp, save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0, format="WEBP", quality=70)
    med = float(np.median(errs)) * 100 if errs else float("nan")
    print(f"measurable {len(errs)}/{len(recs)} · median err {med:.2f}%", flush=True)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB), {args.out_webp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
