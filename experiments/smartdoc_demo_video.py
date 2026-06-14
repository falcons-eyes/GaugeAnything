"""SmartDoc 동적 데모 영상 — 핸드헬드 영상에서 프레임마다 promptable 문서 측정.

P2-1b(smartdoc_detected_quad_eval)의 검출-quad-스케일 파이프라인을 한 시퀀스의
연속 프레임에 적용해, 카메라가 움직이는 동안 문서 quad + A4 치수(mm)가 추종하는
동영상을 만든다. project page용 동적 데모.

각 프레임 오버레이:
  - SAM3 'document' 마스크 → 4코너 quad (게이트 통과 시)
  - detected-quad homography로 GT 코너를 metric 평면에 사상해 변 길이(mm) 측정
  - 측정 A4 폭/높이 mm + GT(210x297) 대비 오차, mm/px 스케일, 프레임 인덱스

Spark 실행:
    .venv/bin/python experiments/smartdoc_demo_video.py \
        --seq background01/datasheet001 --stride 3 --max-frames 70 --fps 12
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


def load_seq_meta(seq: str) -> dict[str, dict]:
    """시퀀스(background/doc)의 프레임별 GT quad 메타."""
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


def measure_quad(det: np.ndarray, gt: np.ndarray, w_mm: float, h_mm: float):
    """detected quad homography로 GT 코너 → mm 변 길이. (top,bottom,left,right) mm + mm/px."""
    import cv2

    dst = np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], np.float32)
    H = cv2.getPerspectiveTransform(det, dst)
    g = cv2.perspectiveTransform(gt.reshape(1, 4, 2), H).reshape(4, 2)
    tl, tr, br, bl = g
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    # px 폭 평균으로 mm/px 근사
    px_top = np.linalg.norm(det[1] - det[0])
    mm_per_px = w_mm / max(px_top, 1.0)
    return (top + bottom) / 2, (left + right) / 2, mm_per_px


def draw(frame: np.ndarray, quad, meas, idx, n, gate_ok) -> np.ndarray:
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    # 상단 HUD 바
    cv2.rectangle(out, (0, 0), (w, 78), (14, 22, 32), -1)
    cv2.putText(out, "GaugeAnything  -  promptable document scale", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (54, 224, 176), 2, cv2.LINE_AA)
    cv2.putText(out, f"prompt: document   |   SAM 3 + metrology core   |   frame {idx+1}/{n}",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215, 230, 239), 1, cv2.LINE_AA)
    if gate_ok and quad is not None:
        q = quad.astype(np.int32)
        cv2.polylines(out, [q], True, (54, 224, 176), 3, cv2.LINE_AA)
        for p in q:
            cv2.circle(out, tuple(p), 6, (34, 195, 230), -1, cv2.LINE_AA)
        wmm, hmm, mmpp = meas
        # 하단 측정 패널
        cv2.rectangle(out, (0, h - 74), (w, h), (14, 22, 32), -1)
        we, he = abs(wmm - A4_W_MM) / A4_W_MM * 100, abs(hmm - A4_H_MM) / A4_H_MM * 100
        cv2.putText(out, f"measured  W {wmm:5.1f}mm (A4 210, {we:.1f}%)   H {hmm:5.1f}mm (A4 297, {he:.1f}%)",
                    (16, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"scale {mmpp:.4f} mm/px   |   homography from detected quad",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (181, 200, 214), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(out, (0, h - 40), (w, h), (40, 22, 26), -1)
        cv2.putText(out, "document not measurable this frame (gate rejected) - reported, not guessed",
                    (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 120, 120), 1, cv2.LINE_AA)
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

    print(f"=== SmartDoc demo video: {args.seq} ===", flush=True)
    meta = load_seq_meta(args.seq)
    frames_sorted = sorted(meta)
    picked = frames_sorted[:: args.stride][: args.max_frames]
    print(f"sequence frames {len(frames_sorted)} → picked {len(picked)} (stride {args.stride})", flush=True)
    images = extract_seq_images(picked)
    print(f"decoded {len(images)} frames", flush=True)

    rendered, gate_pass, errs = [], 0, []
    for i, path in enumerate(picked):
        img = images.get(path)
        row = meta[path]
        if img is None:
            continue
        w_mm = float(row.get("model_width") or 2100.0) * 0.1 or A4_W_MM
        h_mm = float(row.get("model_height") or 2970.0) * 0.1 or A4_H_MM
        quad, meas, ok = None, None, False
        try:
            inst = segment_sam3(img, PROMPT)
            if inst:
                best = max(inst, key=lambda x: x.score)
                quad, why = mask_to_quad(best.mask)
                if quad is not None:
                    gt = gt_quad(row)
                    wmm, hmm, mmpp = measure_quad(quad, gt, w_mm, h_mm)
                    meas, ok = (wmm, hmm, mmpp), True
                    gate_pass += 1
                    errs.append(abs(wmm - w_mm) / w_mm)
        except Exception as e:
            print(f"  frame {i}: {type(e).__name__}", flush=True)
        rendered.append(draw(img, quad, meas, i, len(picked), ok))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(picked)} (gate pass {gate_pass})", flush=True)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not rendered:
        print("no frames rendered", flush=True)
        return 1
    h, w = rendered[0].shape[:2]
    # mp4: 원시 프레임을 PNG로 쓴 뒤 시스템 ffmpeg로 H.264 인코딩 (web 호환 yuv420p)
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i, r in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", r)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
             "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out_mp4)],
            check=True)
    # web-friendly animated webp (절반 크기, 루프) — PIL
    from PIL import Image
    small = [Image.fromarray(cv2.resize(r, (w // 2, h // 2))[:, :, ::-1]) for r in rendered]
    small[0].save(args.out_webp, save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0, format="WEBP", quality=70)
    med = float(np.median(errs)) if errs else float("nan")
    print(f"gate pass {gate_pass}/{len(picked)} · median W err {med*100:.1f}%", flush=True)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB), "
          f"{args.out_webp} ({args.out_webp.stat().st_size//1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
