"""T-LESS 산업부품 치수 측정 데모 영상 (real RGB 시퀀스 + BOP CAD GT).

한 scene의 연속 RGB 프레임마다 SAM3 'plastic part' 검출 → 마스크 최대 chord(px) →
plane-scale(Z/fx)로 mm 치수. CAD+pose에서 유도한 GT mm와 비교 오버레이. 카메라/장면이
바뀌어도 promptable 측정이 mm로 유지됨을 보여준다 (다른 task: 산업부품 dimension).

Spark 실행:
    .venv/bin/python experiments/tless_demo_video.py --scene 000005 --max-frames 36 --fps 10
출력: docs/assets/demos/part_tless_<scene>.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.tless_upper_bound import TLESS, load_ply_vertices, max_chord_2d  # noqa: E402

GREEN, AMBER, CY, GREY, WHITE = (54, 224, 176), (40, 170, 240), (34, 195, 230), (181, 200, 214), (255, 255, 255)
PROMPT = "plastic part"


def mask_iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return i / max(u, 1)


def draw(img, parts, idx, n):
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 70), (14, 22, 32), -1)
    cv2.putText(out, "Promptable part measurement - dimension in millimetres", (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    cv2.putText(out, f"T-LESS RGB sequence  |  prompt: plastic part  |  mask -> max chord -> mm (plane scale)  |  CAD ground truth  |  frame {idx+1}/{n}",
                (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    for p in parts:
        col = GREEN
        cv2.polylines(out, [p["hull"]], True, col, 2, cv2.LINE_AA)
        (x1, y1) = p["chord"][0]; (x2, y2) = p["chord"][1]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), CY, 2, cv2.LINE_AA)
        tx, ty = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.putText(out, f"{p['pred_mm']:.1f}mm", (tx - 28, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
        cv2.putText(out, f"{p['pred_mm']:.1f}mm", (tx - 28, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    # 하단 요약
    if parts:
        errs = [p["rel"] for p in parts if p["rel"] is not None]
        me = np.median(errs) * 100 if errs else float("nan")
        cv2.rectangle(out, (0, h - 34), (w, h), (14, 22, 32), -1)
        cv2.putText(out, f"{len(parts)} parts measured this frame  |  median error vs CAD {me:.1f}%",
                    (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)
    return out


def main() -> int:
    import cv2
    from PIL import Image
    from gaugeanything.segmenters import segment_sam3

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="000005")
    ap.add_argument("--max-frames", type=int, default=36)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--visib", type=float, default=0.6)
    ap.add_argument("--iou-match", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out-mp4", type=Path, default=None)
    args = ap.parse_args()

    sd = TLESS / "test_primesense" / args.scene
    out_mp4 = args.out_mp4 or Path(f"docs/assets/demos/part_tless_{args.scene}.mp4")
    out_webp = out_mp4.with_suffix(".webp")
    models_dir = TLESS / "models_cad"
    cam = json.loads((sd / "scene_camera.json").read_text())
    gt = json.loads((sd / "scene_gt.json").read_text())
    gti = json.loads((sd / "scene_gt_info.json").read_text())
    im_ids = sorted(gt.keys(), key=int)[:: args.stride][: args.max_frames]
    print(f"=== T-LESS demo: scene {args.scene}, {len(im_ids)} frames ===", flush=True)

    model_cache: dict[int, np.ndarray] = {}
    rendered, all_err = [], []
    for fi, im in enumerate(im_ids):
        rgb = np.array(Image.open(sd / "rgb" / f"{int(im):06d}.png").convert("RGB"))
        K = np.array(cam[im]["cam_K"]).reshape(3, 3); fx = K[0, 0]
        # GT 객체: visible mask + gt_mm + tz + centroid
        gts = []
        for gi, (obj, info) in enumerate(zip(gt[im], gti[im])):
            if info.get("visib_fract", 0) < args.visib:
                continue
            oid = obj["obj_id"]
            if oid not in model_cache:
                model_cache[oid] = load_ply_vertices(models_dir / f"obj_{oid:06d}.ply")
            V = model_cache[oid]
            R = np.array(obj["cam_R_m2c"]).reshape(3, 3); t = np.array(obj["cam_t_m2c"]).reshape(3)
            Xc = V @ R.T + t
            P = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
            _, i, j = max_chord_2d(P)
            gt_mm = float(np.linalg.norm(Xc[i] - Xc[j]))
            mp = sd / "mask_visib" / f"{int(im):06d}_{gi:06d}.png"
            if not mp.exists():
                continue
            m = np.array(Image.open(mp)) > 0
            gts.append({"mask": m, "gt_mm": gt_mm, "tz": float(t[2])})
        # SAM3 promptable
        parts = []
        try:
            inst = segment_sam3(rgb, PROMPT)
        except Exception as e:
            print(f"  frame {fi}: {type(e).__name__}", flush=True); inst = []
        for s in inst:
            sm = s.mask.astype(bool)
            if sm.sum() < 80:
                continue
            best, biou = None, 0
            for g in gts:
                iou = mask_iou(sm, g["mask"])
                if iou > biou:
                    biou, best = iou, g
            if best is None or biou < args.iou_match:
                continue
            ys, xs = np.nonzero(sm)
            pts = np.stack([xs, ys], 1).astype(np.float64)
            chord_px, ci, cj = max_chord_2d(pts)
            pred_mm = chord_px * best["tz"] / fx
            rel = abs(pred_mm - best["gt_mm"]) / max(best["gt_mm"], 1e-6)
            all_err.append(rel)
            try:
                import cv2
                hull = cv2.convexHull(pts.astype(np.int32))
            except Exception:
                hull = pts.astype(np.int32).reshape(-1, 1, 2)
            parts.append({"hull": hull, "chord": (pts[ci], pts[cj]), "pred_mm": pred_mm, "rel": rel})
        rendered.append(draw(rgb, parts, fi, len(im_ids)))
        if (fi + 1) % 10 == 0:
            print(f"  {fi+1}/{len(im_ids)} ({len(parts)} parts)", flush=True)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im[:, :, ::-1])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)], check=True)
    h, w = rendered[0].shape[:2]
    small = [Image.fromarray(cv2.resize(im, (w // 2, h // 2))) for im in rendered]
    small[0].save(out_webp, save_all=True, append_images=small[1:], duration=int(1000 / args.fps), loop=0, format="WEBP", quality=72)
    me = np.median(all_err) * 100 if all_err else float("nan")
    print(f"measured {len(all_err)} part-instances, median error {me:.1f}% vs CAD", flush=True)
    print(f"wrote {out_mp4} ({out_mp4.stat().st_size//1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
