"""크랙 폭 Ken Burns 의사영상 — 고해상 스캔을 따라 뷰가 이동하며 폭을 측정.

⚠️ 합성 모션: 정지 이미지(krkCMd 9448x6305 콘크리트 스캔, CC BY 4.0, GT 폭 파일명 인코딩)에
가상 카메라(Ken Burns) 팬·줌을 준다. 실제 비디오가 아님을 화면에 명시한다.
각 위치에서 크랙 중심선의 국소 폭(2×EDT)을 측정하고, 데이터셋 GT 평균(예: 0.23mm)에
스케일을 캘리브레이션해 mm로 표시한다. 폭이 위치마다 변하는 것을 보여준다.

Spark 실행:
    .venv/bin/python experiments/crack_kenburns_video.py \
        --scan-glob 'datasets/krkcmd/**/CMd_0.23*Image1.tif' --gt-mm 0.23 \
        --frames 80 --fps 14
출력: docs/assets/crack_demo.mp4, docs/assets/crack_demo.webp
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

OUT_MP4 = Path("docs/assets/crack_demo.mp4")
OUT_WEBP = Path("docs/assets/crack_demo.webp")
VIEW_W, VIEW_H = 900, 560     # 출력 뷰포트
ZOOM = 2.2                    # Ken Burns 확대 배율


def find_crack_centerline(gray: np.ndarray):
    """어두운 크랙 → 마스크 → 스켈레톤 → 최장-스켈레톤 성분의 x-정렬 경로 + EDT.

    스캔 테두리(검은 경계)는 큰 면적 성분이라 area 최대로는 잡힌다 → 경계 접촉 성분
    제거 + '스켈레톤 길이 최대'(가장 크랙다운, 길고 가는 구조)로 선택.
    """
    import cv2
    from scipy import ndimage
    from skimage.morphology import skeletonize

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 스캔 검은 테두리 strip 제거: 외곽 마진 밴드를 0으로 (크랙 내부는 보존)
    binimg = (thr > 0)
    Hc, Wc = binimg.shape
    mx, my = int(Wc * 0.04), int(Hc * 0.04)
    binimg[:my] = binimg[-my:] = False
    binimg[:, :mx] = binimg[:, -mx:] = False
    lab, n = ndimage.label(binimg)
    if n == 0:
        raise RuntimeError("no crack mask after border clear")
    # 성분별 스켈레톤 길이로 크랙 선택 (area 아님)
    best_len, best_k = -1, -1
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() < 200:
            continue
        sl = int(skeletonize(comp).sum())
        if sl > best_len:
            best_len, best_k = sl, k
    if best_k < 0:
        raise RuntimeError("no crack-like component")
    mask = lab == best_k
    edt = ndimage.distance_transform_edt(mask)
    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    order = np.argsort(xs)         # 대략 수평 크랙 → x로 진행 순서
    return mask, edt, np.stack([xs[order], ys[order]], 1)


def smooth_path(pts: np.ndarray, n: int) -> np.ndarray:
    """경로를 n개 균등 샘플 + 이동평균으로 부드럽게."""
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    p = pts[idx].astype(float)
    k = 5
    for c in range(2):
        p[:, c] = np.convolve(p[:, c], np.ones(k) / k, mode="same")
    return p


def draw_view(scan_bgr, edt, cx, cy, width_mm, gt_mm, idx, n, scale_mm_px, path_full):
    import cv2

    H, W = scan_bgr.shape[:2]
    half_w, half_h = int(VIEW_W / ZOOM / 2), int(VIEW_H / ZOOM / 2)
    x0, y0 = int(np.clip(cx - half_w, 0, W - 2 * half_w)), int(np.clip(cy - half_h, 0, H - 2 * half_h))
    crop = scan_bgr[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w]
    view = cv2.resize(crop, (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)
    vx, vy = int((cx - x0) * ZOOM), int((cy - y0) * ZOOM)
    # 측정 크로스헤어 + 폭 막대
    cv2.line(view, (vx - 26, vy), (vx + 26, vy), (54, 224, 176), 2, cv2.LINE_AA)
    cv2.line(view, (vx, vy - 26), (vx, vy + 26), (54, 224, 176), 2, cv2.LINE_AA)
    wpx = width_mm / scale_mm_px * ZOOM
    cv2.line(view, (vx, int(vy - wpx / 2)), (vx, int(vy + wpx / 2)), (34, 195, 230), 4, cv2.LINE_AA)
    # HUD
    cv2.rectangle(view, (0, 0), (VIEW_W, 64), (14, 22, 32), -1)
    cv2.putText(view, "GaugeAnything  -  crack width, measured along the crack", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (54, 224, 176), 2, cv2.LINE_AA)
    cv2.putText(view, f"krkCMd scanner image (CC BY 4.0)   |   synthetic Ken Burns pan over a static 9448x6305 scan   |   {idx+1}/{n}",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 230, 239), 1, cv2.LINE_AA)
    # 측정 패널
    cv2.rectangle(view, (0, VIEW_H - 58), (VIEW_W, VIEW_H), (14, 22, 32), -1)
    cv2.putText(view, f"local width {width_mm*1000:5.0f} um   ({width_mm:.3f} mm)   |   dataset GT mean {gt_mm:.2f} mm",
                (14, VIEW_H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "width varies along the crack; scale calibrated to the GT mean",
                (14, VIEW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (181, 200, 214), 1, cv2.LINE_AA)
    # 미니맵 (전체 크랙에서 현재 위치)
    mm_w, mm_h = 220, int(220 * H / W)
    mini = cv2.resize(scan_bgr, (mm_w, mm_h))
    cv2.polylines(mini, [(path_full * [mm_w / W, mm_h / H]).astype(np.int32)], False, (54, 224, 176), 1, cv2.LINE_AA)
    cv2.circle(mini, (int(cx * mm_w / W), int(cy * mm_h / H)), 4, (34, 195, 230), -1, cv2.LINE_AA)
    view[VIEW_H - 58 - mm_h - 8:VIEW_H - 58 - 8, VIEW_W - mm_w - 8:VIEW_W - 8] = mini
    return view


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-glob", default="datasets/krkcmd/**/CMd_0.23*Image1.tif")
    ap.add_argument("--gt-mm", type=float, default=0.23)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--proc-width", type=int, default=3600, help="처리용 다운스케일 폭")
    ap.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    ap.add_argument("--out-webp", type=Path, default=OUT_WEBP)
    args = ap.parse_args()

    files = sorted(glob.glob(args.scan_glob, recursive=True))
    if not files:
        print(f"no scan for {args.scan_glob}")
        return 1
    print(f"=== crack Ken Burns: {files[0].split('/')[-1]} ===", flush=True)
    from PIL import Image
    full = np.array(Image.open(files[0]).convert("L"))
    H0, W0 = full.shape
    s = args.proc_width / W0
    gray = cv2.resize(full, (args.proc_width, int(H0 * s)), interpolation=cv2.INTER_AREA)
    scan_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(f"scan {W0}x{H0} → proc {gray.shape[1]}x{gray.shape[0]}", flush=True)

    mask, edt, path = find_crack_centerline(gray)
    path = smooth_path(path, args.frames)
    # 각 위치 국소 폭 = 중심점 주변 EDT 최대 ×2 (px)
    widths_px = []
    for x, y in path:
        xi, yi = int(x), int(y)
        patch = edt[max(0, yi - 6):yi + 7, max(0, xi - 6):xi + 7]
        widths_px.append(2 * float(patch.max()) if patch.size else 0.0)
    widths_px = np.array(widths_px)
    # 0/결측(브랜치 갭) 폭은 유효 이웃으로 보간 — "0 um" 프레임 방지
    valid = widths_px > 0
    if valid.any() and not valid.all():
        xs_idx = np.arange(len(widths_px))
        widths_px = np.interp(xs_idx, xs_idx[valid], widths_px[valid])
    med = np.median(widths_px[widths_px > 0])
    scale_mm_px = args.gt_mm / max(med, 1e-6)     # GT 평균에 캘리브레이션
    widths_mm = widths_px * scale_mm_px
    print(f"path {len(path)} pts · median width {med:.1f}px → scale {scale_mm_px*1000:.2f} um/px", flush=True)

    rendered = []
    for i, (x, y) in enumerate(path):
        rendered.append(draw_view(scan_bgr, edt, x, y, widths_mm[i], args.gt_mm, i, len(path),
                                  scale_mm_px, path))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out_mp4)], check=True)
    small = [Image.fromarray(cv2.resize(im, (VIEW_W // 2, VIEW_H // 2))[:, :, ::-1]) for im in rendered]
    small[0].save(args.out_webp, save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0, format="WEBP", quality=72)
    print(f"width range {widths_mm.min()*1000:.0f}-{widths_mm.max()*1000:.0f}um", flush=True)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB), {args.out_webp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
