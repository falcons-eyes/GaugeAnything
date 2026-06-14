"""크랙 폭 데모 — "mask로 재면 틀린다, signal로 재야 한다" (우리 논문의 중심 발견).

⚠️ 합성 모션: 정지 고해상 스캔(krkCMd 9448x6305, CC BY 4.0, GT 폭 0.23mm)에 가상 카메라
(Ken Burns)가 크랙을 따라 이동. 실제 비디오 아님을 화면에 명시.

각 측정점에서 크랙에 수직인 밝기 단면(profile)을 뽑아 두 가지로 폭을 잰다:
  - BINARY MASK width: 이진 마스크의 span. 점진적 경계(penumbra)까지 1로 잡아 부풀려진다(틀림).
  - SIGNAL width: 밝기 골짜기의 half-depth 폭. 서브픽셀, GT에 정합(정답).
단면 그래프에 두 폭을 겹쳐 그려 "왜 마스크로는 안 되는지"를 가시화한다. scale은 signal 중앙값을
GT 평균에 캘리브레이션.

Spark 실행:
    .venv/bin/python experiments/crack_kenburns_video.py --gt-mm 0.23 --frames 80 --fps 14
출력: docs/assets/crack_demo.mp4, docs/assets/crack_demo.webp
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

OUT_MP4 = Path("docs/assets/crack_demo.mp4")
OUT_WEBP = Path("docs/assets/crack_demo.webp")
VIEW_W, VIEW_H = 1000, 620
ZOOM = 2.4
NORM_HALF = 46          # 단면 샘플 반경(px, proc 스케일)
GREEN, AMBER, CY, GREY, WHITE = (54, 224, 176), (40, 170, 240), (34, 195, 230), (181, 200, 214), (255, 255, 255)


def find_crack(gray):
    import cv2
    from scipy import ndimage
    from skimage.morphology import skeletonize

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binimg = thr > 0
    Hc, Wc = binimg.shape
    mx, my = int(Wc * 0.04), int(Hc * 0.04)
    binimg[:my] = binimg[-my:] = False
    binimg[:, :mx] = binimg[:, -mx:] = False
    lab, n = ndimage.label(binimg)
    if n == 0:
        raise RuntimeError("no crack")
    best_len, best_k = -1, -1
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() < 200:
            continue
        sl = int(skeletonize(comp).sum())
        if sl > best_len:
            best_len, best_k = sl, k
    mask = lab == best_k
    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    order = np.argsort(xs)
    return mask, np.stack([xs[order], ys[order]], 1)


def smooth_path(pts, n):
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    p = pts[idx].astype(float)
    for c in range(2):
        p[:, c] = np.convolve(p[:, c], np.ones(5) / 5, mode="same")
    return p


def sample_normal(gray, mask, cx, cy, tangent):
    """크랙 수직 방향 밝기 단면 + 마스크 단면. 반환: (offsets, intensity, mask_line)."""
    import cv2
    nx, ny = -tangent[1], tangent[0]                  # 법선
    norm = np.hypot(nx, ny) or 1.0
    nx, ny = nx / norm, ny / norm
    offs = np.arange(-NORM_HALF, NORM_HALF + 1)
    xs = cx + offs * nx
    ys = cy + offs * ny
    inten = cv2.remap(gray, xs.astype(np.float32).reshape(-1, 1), ys.astype(np.float32).reshape(-1, 1),
                      cv2.INTER_LINEAR).ravel().astype(float)
    mline = cv2.remap(mask.astype(np.uint8) * 255, xs.astype(np.float32).reshape(-1, 1),
                      ys.astype(np.float32).reshape(-1, 1), cv2.INTER_NEAREST).ravel() > 127
    return offs, inten, mline


def measure_widths(offs, inten, mline):
    """mask 폭(span) + signal 폭(밝기 골짜기 half-depth). px."""
    mask_w = float(mline.sum())
    bg = np.median(np.concatenate([inten[:12], inten[-12:]]))
    vmin = inten.min()
    if bg - vmin < 8:                                 # 대비 부족 → 측정 불가
        return mask_w, np.nan, bg, vmin, None
    half = (bg + vmin) / 2
    below = inten < half
    if not below.any():
        return mask_w, np.nan, bg, vmin, None
    idxs = np.where(below)[0]
    sig_w = float(offs[idxs[-1]] - offs[idxs[0]] + 1)
    return mask_w, sig_w, bg, vmin, (offs[idxs[0]], offs[idxs[-1]], half)


def draw_view(scan_bgr, cx, cy, mask_mm, sig_mm, gt_mm, prof, idx, n, scale_mm_px, path_full):
    import cv2
    H, W = scan_bgr.shape[:2]
    hw, hh = int(VIEW_W / ZOOM / 2), int(VIEW_H / ZOOM / 2)
    x0 = int(np.clip(cx - hw, 0, W - 2 * hw)); y0 = int(np.clip(cy - hh, 0, H - 2 * hh))
    view = cv2.resize(scan_bgr[y0:y0 + 2 * hh, x0:x0 + 2 * hw], (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)
    vx, vy = int((cx - x0) * ZOOM), int((cy - y0) * ZOOM)
    # signal 폭 막대 (green) — raw 프로파일에서 읽은 sub-pixel 폭
    if not np.isnan(sig_mm):
        wpx = sig_mm / scale_mm_px * ZOOM
        cv2.line(view, (vx, int(vy - wpx / 2)), (vx, int(vy + wpx / 2)), GREEN, 3, cv2.LINE_AA)
    cv2.circle(view, (vx, vy), 3, CY, -1, cv2.LINE_AA)

    # HUD
    cv2.rectangle(view, (0, 0), (VIEW_W, 92), (14, 22, 32), -1)
    cv2.putText(view, "Width read from the raw signal - a physical quantity, not a box.", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, GREEN, 2, cv2.LINE_AA)
    cv2.putText(view, f"krkCMd scan (CC BY 4.0)  |  synthetic Ken Burns over a static 9448x6305 image  |  a {gt_mm:.2f} mm crack is ~50 px wide  |  {idx+1}/{n}",
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)

    # 단면 프로파일 패널 (우상단)
    pw, ph, px, py = 360, 200, VIEW_W - 360 - 16, 104
    cv2.rectangle(view, (px, py), (px + pw, py + ph), (10, 16, 24), -1)
    cv2.rectangle(view, (px, py), (px + pw, py + ph), (40, 54, 68), 1)
    cv2.putText(view, "cross-section brightness (the crack is a valley, not an edge)", (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    if prof is not None:
        offs, inten, mline, span, bg, vmin = prof
        def Y(v): return py + 34 + int((bg - v) / max(bg - vmin, 1) * (ph - 54))
        def X(o): return px + 10 + int((o - offs[0]) / max(offs[-1] - offs[0], 1) * (pw - 20))
        # signal half-depth 폭 (green 막대 + 가이드)
        if span is not None:
            a, b, half = span
            yy = Y(half)
            cv2.line(view, (X(a), yy), (X(b), yy), GREEN, 2, cv2.LINE_AA)
            for xx in (X(a), X(b)):
                cv2.line(view, (xx, yy - 5), (xx, yy + 5), GREEN, 2, cv2.LINE_AA)
            cv2.putText(view, "width @ half-depth", (X(a), yy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, GREEN, 1, cv2.LINE_AA)
        # 밝기 곡선
        pts = [[X(o), Y(v)] for o, v in zip(offs, inten)]
        cv2.polylines(view, [np.array(pts, np.int32)], False, WHITE, 1, cv2.LINE_AA)

    # 측정 패널 (하단)
    cv2.rectangle(view, (0, VIEW_H - 70), (VIEW_W, VIEW_H), (14, 22, 32), -1)
    if not np.isnan(sig_mm):
        cv2.putText(view, f"crack width  {sig_mm*1000:4.0f} um   ({sig_mm:.3f} mm)", (16, VIEW_H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREEN, 2, cv2.LINE_AA)
        cv2.putText(view, f"sub-pixel, from the brightness profile   |   GT mean {gt_mm*1000:.0f} um, width varies along the crack",
                    (16, VIEW_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREY, 1, cv2.LINE_AA)
    else:
        cv2.putText(view, "low contrast here - not measurable (reported, not guessed)", (16, VIEW_H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)

    # 미니맵
    mm_w, mm_h = 200, int(200 * H / W)
    mini = cv2.resize(scan_bgr, (mm_w, mm_h))
    cv2.polylines(mini, [(path_full * [mm_w / W, mm_h / H]).astype(np.int32)], False, GREEN, 1, cv2.LINE_AA)
    cv2.circle(mini, (int(cx * mm_w / W), int(cy * mm_h / H)), 4, CY, -1, cv2.LINE_AA)
    view[VIEW_H - 70 - mm_h - 8:VIEW_H - 70 - 8, VIEW_W - mm_w - 8:VIEW_W - 8] = mini
    return view


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-glob", default="datasets/krkcmd/**/CMd_0.23*Image1.tif")
    ap.add_argument("--gt-mm", type=float, default=0.23)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--proc-width", type=int, default=3600)
    ap.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    ap.add_argument("--out-webp", type=Path, default=OUT_WEBP)
    args = ap.parse_args()

    files = sorted(glob.glob(args.scan_glob, recursive=True))
    if not files:
        print(f"no scan for {args.scan_glob}"); return 1
    print(f"=== crack mask-vs-signal: {files[0].split('/')[-1]} ===", flush=True)
    from PIL import Image
    full = np.array(Image.open(files[0]).convert("L"))
    H0, W0 = full.shape
    s = args.proc_width / W0
    gray = cv2.resize(full, (args.proc_width, int(H0 * s)), interpolation=cv2.INTER_AREA)
    scan_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask, path = find_crack(gray)
    path = smooth_path(path, args.frames)
    print(f"scan {W0}x{H0} -> {gray.shape[1]}x{gray.shape[0]}, path {len(path)}", flush=True)

    # 각 측정점: tangent → normal 단면 → mask/signal 폭
    recs = []
    for i, (cx, cy) in enumerate(path):
        a = path[max(0, i - 2)]; b = path[min(len(path) - 1, i + 2)]
        tangent = (b - a)
        if np.hypot(*tangent) < 1e-3:
            tangent = np.array([1.0, 0.0])
        offs, inten, mline = sample_normal(gray, mask, cx, cy, tangent)
        mask_w, sig_w, bg, vmin, span = measure_widths(offs, inten, mline)
        recs.append({"cx": cx, "cy": cy, "mask_px": mask_w, "sig_px": sig_w,
                     "prof": (offs, inten, mline, span, bg, vmin)})

    sig_all = np.array([r["sig_px"] for r in recs if not np.isnan(r["sig_px"])])
    mask_all = np.array([r["mask_px"] for r in recs if r["mask_px"] > 0])
    scale_mm_px = args.gt_mm / max(np.median(sig_all), 1e-6)     # signal 중앙값 → GT 캘리브
    ratio = np.median(mask_all) / max(np.median(sig_all), 1e-6)
    print(f"signal median {np.median(sig_all):.1f}px, mask median {np.median(mask_all):.1f}px → mask/signal {ratio:.1f}x", flush=True)

    rendered = []
    for i, r in enumerate(recs):
        mm = r["mask_px"] * scale_mm_px if r["mask_px"] > 0 else np.nan
        sm = r["sig_px"] * scale_mm_px if not np.isnan(r["sig_px"]) else np.nan
        rendered.append(draw_view(scan_bgr, r["cx"], r["cy"], mm, sm, args.gt_mm, r["prof"],
                                  i, len(recs), scale_mm_px, path))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out_mp4)], check=True)
    sm = [Image.fromarray(cv2.resize(im, (VIEW_W // 2, VIEW_H // 2))[:, :, ::-1]) for im in rendered]
    sm[0].save(args.out_webp, save_all=True, append_images=sm[1:],
               duration=int(1000 / args.fps), loop=0, format="WEBP", quality=72)
    print(f"mask/signal width ratio {ratio:.1f}x (the cost of reading width from a mask)", flush=True)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB), {args.out_webp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
