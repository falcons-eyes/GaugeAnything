"""SmartDoc 동적 데모 영상 v2 — "detection으로는 안 되고, metric 측정이 필요하다".

핵심 메시지: 같은 detected quad를 써도, 픽셀 크기로 재는 순진한 방법(=detection+tracking이
줄 수 있는 것)은 카메라가 움직이면 10-17% 틀린다. 우리의 per-frame homography metric만이
A4 mm를 0.7%로 잡는다. 두 측정을 나란히 + 발산하는 오차 sparkline + metric rectified 인셋으로
"우리가 하는 건 추적이 아니라 정밀 계측"을 가시화한다.

  - naive scale: 첫 프레임에서 mm/px 고정(fronto-parallel 가정) → 매 프레임 apparent px 크기로 환산.
    카메라 거리/틸트가 바뀌면 출렁인다. (detection+tracking only)
  - metric: per-frame homography로 metric 평면 복원 → 진짜 A4 mm. (GaugeAnything)
  - rectified 인셋: detected quad를 210x297 평면으로 warp → "기하를 푸는 중"임을 증명.

Spark 실행:
    .venv/bin/python experiments/smartdoc_demo_video.py --stride 3 --max-frames 70 --fps 12
출력: docs/assets/smartdoc_demo.mp4, docs/assets/smartdoc_demo.webp
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.smartdoc_detected_quad_eval import (  # noqa: E402
    A4_H_MM, A4_W_MM, mask_to_quad, order_corners,
)

DATA = Path("datasets/smartdoc")
OUT_MP4 = Path("docs/assets/smartdoc_demo.mp4")
OUT_WEBP = Path("docs/assets/smartdoc_demo.webp")
PROMPT = "document"
GREEN, AMBER, RED, CY, GREY, WHITE = (54, 224, 176), (40, 170, 240), (90, 90, 230), (34, 195, 230), (181, 200, 214), (255, 255, 255)


def load_seq_meta(seq: str) -> dict[str, dict]:
    import csv
    tf = tarfile.open(DATA / "frames.tar.gz")
    m = next(x for x in tf.getmembers() if x.name.endswith("metadata.csv.gz"))
    payload = tf.extractfile(m).read()
    tf.close()
    rows = {}
    for r in csv.DictReader(gzip.open(io.BytesIO(payload), "rt", encoding="utf-8", newline="")):
        if r["image_path"].startswith(seq + "/"):
            rows[r["image_path"]] = r
    return rows


def extract_seq_images(paths: list[str]) -> dict[str, np.ndarray]:
    from PIL import Image
    want = set(paths)
    out = {}
    with tarfile.open(DATA / "frames.tar.gz") as tf:
        for mem in tf:
            if mem.name in want:
                out[mem.name] = np.array(Image.open(io.BytesIO(tf.extractfile(mem).read())).convert("RGB"))
                if len(out) == len(want):
                    break
    return out


def gt_quad(row: dict) -> np.ndarray:
    return order_corners(np.array(
        [[float(row["tl_x"]), float(row["tl_y"])], [float(row["tr_x"]), float(row["tr_y"])],
         [float(row["br_x"]), float(row["br_y"])], [float(row["bl_x"]), float(row["bl_y"])]],
        np.float32))


def edges_px(q: np.ndarray):
    tl, tr, br, bl = q
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    return top, bottom, left, right


def metric_measure(det: np.ndarray, gt: np.ndarray, w_mm, h_mm):
    """per-frame homography로 GT 코너 → mm. 진짜 metric 측정."""
    import cv2
    dst = np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], np.float32)
    H = cv2.getPerspectiveTransform(det, dst)
    g = cv2.perspectiveTransform(gt.reshape(1, 4, 2), H).reshape(4, 2)
    t, b, l, r = edges_px(g)
    return (t + b) / 2, (l + r) / 2, H


def tilt_estimate(det: np.ndarray) -> float:
    """detected quad의 원근 단축 → 대략적 카메라 틸트(도). 시각 지표용."""
    t, b, l, r = edges_px(det)
    fore = max(max(t, b) / max(min(t, b), 1), max(l, r) / max(min(l, r), 1))  # ≥1
    return float(np.degrees(np.arccos(min(1.0, 1.0 / fore))))


def draw_spark(canvas, x0, y0, w, h, naive_err, ours_err, idx):
    import cv2
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (10, 16, 24), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (40, 54, 68), 1)
    ymax = 20.0  # %
    def yat(e): return y0 + h - int(np.clip(e / ymax, 0, 1) * (h - 6)) - 3
    for gl, lab in [(0, "0%"), (10, "10%"), (20, "20%")]:
        yy = yat(gl)
        cv2.line(canvas, (x0, yy), (x0 + w, yy), (30, 42, 54), 1)
        cv2.putText(canvas, lab, (x0 + 3, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 110, 126), 1, cv2.LINE_AA)
    m = max(1, idx + 1)
    def trace(errs, col):
        pts = []
        for i in range(m):
            xx = x0 + int(i / max(1, len(errs) - 1) * (w - 6)) + 3
            pts.append([xx, yat(errs[i])])
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    trace(naive_err, AMBER)
    trace(ours_err, GREEN)
    cv2.putText(canvas, "measurement error vs reference  (lower is better)", (x0 + 6, y0 + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREY, 1, cv2.LINE_AA)


def draw(frame, quad, rec, hist, idx, n):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    # 상단 HUD
    cv2.rectangle(out, (0, 0), (w, 92), (14, 22, 32), -1)
    cv2.putText(out, "Detection finds the page.  The hard part is reading its size in millimetres.",
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, GREEN, 2, cv2.LINE_AA)
    cv2.putText(out, f"SmartDoc handheld scan   |   prompt: document   |   same detected box, two ways to read its size   |   frame {idx+1}/{n}",
                (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    if quad is None:
        cv2.rectangle(out, (0, h - 40), (w, h), (40, 22, 26), -1)
        cv2.putText(out, "document not measurable this frame (gate rejected) - reported, not guessed",
                    (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
        return out
    q = quad.astype(np.int32)
    cv2.polylines(out, [q], True, GREEN, 3, cv2.LINE_AA)
    for p in q:
        cv2.circle(out, tuple(p), 6, CY, -1, cv2.LINE_AA)
    naive_w, ours_w, tilt = rec["naive_w"], rec["ours_w"], rec["tilt"]
    ref_w = rec["w_mm"]
    ne = abs(naive_w - ref_w) / ref_w * 100
    oe = abs(ours_w - ref_w) / ref_w * 100

    # 우측 비교 패널
    pw, px0 = 470, w - 470 - 24
    py0 = 116
    panel = out[py0:py0 + 250, px0:px0 + pw].copy()
    cv2.rectangle(out, (px0, py0), (px0 + pw, py0 + 250), (12, 18, 28), -1)
    cv2.addWeighted(out[py0:py0 + 250, px0:px0 + pw], 0.82, panel, 0.18, 0, out[py0:py0 + 250, px0:px0 + pw])
    cv2.rectangle(out, (px0, py0), (px0 + pw, py0 + 250), (40, 54, 68), 1)
    # naive
    cv2.putText(out, "DETECTION + TRACKING", (px0 + 16, py0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)
    cv2.putText(out, "(pixel size x fixed scale)", (px0 + 16, py0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    cv2.putText(out, f"{naive_w:5.0f} mm", (px0 + 200, py0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, AMBER, 2, cv2.LINE_AA)
    cv2.putText(out, f"off by {ne:4.1f}%", (px0 + 200, py0 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)
    cv2.line(out, (px0 + 16, py0 + 92), (px0 + pw - 16, py0 + 92), (40, 54, 68), 1)
    # ours
    cv2.putText(out, "GAUGEANYTHING (metric)", (px0 + 16, py0 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.putText(out, "(per-frame homography to a metric plane)", (px0 + 16, py0 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.putText(out, f"{ours_w:5.1f} mm", (px0 + 200, py0 + 138), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, cv2.LINE_AA)
    cv2.putText(out, f"{oe:4.1f}% from true {ref_w:.0f}mm", (px0 + 200, py0 + 164), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.putText(out, f"camera tilt ~{tilt:2.0f} deg   |   why naive drifts: pixels are not millimetres",
                (px0 + 16, py0 + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    # rectified A4 인셋
    if rec.get("rectified") is not None:
        ins = rec["rectified"]
        ih = 224; iw = max(1, int(ih * ins.shape[1] / ins.shape[0]))
        ins = cv2.resize(ins, (iw, ih))
        ix, iy = px0 + 16, py0 + 250 + 16
        out[iy:iy + ih, ix:ix + iw] = ins
        cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), GREEN, 2)
        cv2.putText(out, "metric-rectified document", (ix, iy + ih + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
        cv2.putText(out, "(homography output -", (ix, iy + ih + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)
        cv2.putText(out, " geometry solved, not just a box)", (ix, iy + ih + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)
    # 하단 sparkline
    draw_spark(out, 24, h - 150, 620, 130, hist["naive"][:idx + 1], hist["ours"][:idx + 1], idx)
    return out


def main() -> int:
    import cv2
    from gaugeanything.segmenters import segment_sam3

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="background01/datasheet001")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=70)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    ap.add_argument("--out-webp", type=Path, default=OUT_WEBP)
    args = ap.parse_args()

    print(f"=== SmartDoc demo v2 (naive vs metric): {args.seq} ===", flush=True)
    meta = load_seq_meta(args.seq)
    picked = sorted(meta)[:: args.stride][: args.max_frames]
    images = extract_seq_images(picked)
    print(f"frames {len(picked)}, decoded {len(images)}", flush=True)

    # PASS 1: 측정 수집 (naive + metric)
    recs = []
    naive_mm_per_px = None
    for i, path in enumerate(picked):
        img = images.get(path); row = meta[path]
        rec = {"img": img, "quad": None}
        if img is not None:
            w_mm = float(row.get("model_width") or 2100.0) * 0.1 or A4_W_MM
            h_mm = float(row.get("model_height") or 2970.0) * 0.1 or A4_H_MM
            try:
                inst = segment_sam3(img, PROMPT)
                if inst:
                    quad, _ = mask_to_quad(max(inst, key=lambda x: x.score).mask)
                    if quad is not None:
                        gt = gt_quad(row)
                        ours_w, ours_h, H = metric_measure(quad, gt, w_mm, h_mm)
                        t, b, l, r = edges_px(quad)
                        app_w = (t + b) / 2
                        if naive_mm_per_px is None:    # 첫 측정 프레임에서 스케일 고정
                            naive_mm_per_px = w_mm / max(app_w, 1.0)
                        naive_w = app_w * naive_mm_per_px
                        # rectified 인셋: detected quad → 디스플레이 직사각형 warp (문서 비율 보존)
                        DW = 360; DH = int(DW * h_mm / w_mm)
                        Hd = cv2.getPerspectiveTransform(
                            quad.astype(np.float32),
                            np.array([[0, 0], [DW, 0], [DW, DH], [0, DH]], np.float32))
                        rect = cv2.warpPerspective(img, Hd, (DW, DH))
                        rec.update(quad=quad, naive_w=naive_w, ours_w=ours_w, tilt=tilt_estimate(quad),
                                   rectified=rect, w_mm=w_mm, h_mm=h_mm)
            except Exception as e:
                print(f"  frame {i}: {type(e).__name__}", flush=True)
        recs.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  measured {i+1}/{len(picked)}", flush=True)

    # 오차 히스토리 (sparkline용) — 측정 불가 프레임은 직전값 유지
    hist = {"naive": [], "ours": []}
    ln, lo = 0.0, 0.0
    for r in recs:
        if r["quad"] is not None:
            ln = abs(r["naive_w"] - r["w_mm"]) / r["w_mm"] * 100
            lo = abs(r["ours_w"] - r["w_mm"]) / r["w_mm"] * 100
        hist["naive"].append(ln); hist["ours"].append(lo)

    # PASS 2: 렌더
    rendered = [draw(r["img"], r["quad"], r, hist, i, len(recs)) for i, r in enumerate(recs) if r["img"] is not None]

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
    H, W = rendered[0].shape[:2]
    small = [Image.fromarray(cv2.resize(im, (W // 2, H // 2))[:, :, ::-1]) for im in rendered]
    small[0].save(args.out_webp, save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0, format="WEBP", quality=70)
    nm = [h for h, r in zip(hist["naive"], recs) if r["quad"] is not None]
    om = [h for h, r in zip(hist["ours"], recs) if r["quad"] is not None]
    print(f"naive median {np.median(nm):.1f}% (max {max(nm):.1f}%)  vs  metric median {np.median(om):.1f}%", flush=True)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB), {args.out_webp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
