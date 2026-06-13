"""P2-1b — SmartDoc detected-quad scale: GT quad 상한을 promptable 성능으로.

P2-1(smartdoc_scale_eval.py)은 GT quadrilateral을 썼다 — homography 상한은 정의상 0%다.
이 스크립트는 그 상한을 실제 promptable 성능으로 교체한다:

  frame -> SAM3(prompt) -> document mask -> quadrilateral 추출(gate)
        -> detected-quad homography H_det (image -> A4 metric plane)
        -> GT 코너를 H_det로 사상해 변 길이 측정 -> A4 규격 대비 rel err

게이트(4코너·볼록·면적·mask 정합)를 통과 못 하면 "측정 불가"로 따로 센다 —
coverage와 정확도를 함께 보고하는 것이 계약이다 (benchmark/README.md 공통 규칙 6).

Spark 실행:
    .venv/bin/python -u experiments/smartdoc_detected_quad_eval.py --max-frames 240
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

A4_W_MM = 210.0
A4_H_MM = 297.0
PROMPTS = ["document", "paper", "white paper sheet"]
OUT = Path("experiments/results/smartdoc_detected_quad_eval.json")


def load_metadata(data_root: Path) -> list[dict]:
    archive = data_root / "frames.tar.gz"
    tf = tarfile.open(archive)
    member = next(m for m in tf.getmembers() if m.name.endswith("metadata.csv.gz"))
    payload = tf.extractfile(member).read()
    tf.close()
    import csv

    rows = list(csv.DictReader(gzip.open(io.BytesIO(payload), "rt", encoding="utf-8", newline="")))
    return rows


def stratified_sample(rows: list[dict], n_total: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["bg_name"], r["model_name"])].append(r)
    per_group = max(1, n_total // len(groups))
    picked = []
    for key in sorted(groups):
        g = groups[key]
        step = max(1, len(g) // per_group)
        picked.extend(g[::step][:per_group])
    return picked[:n_total]


def extract_images(data_root: Path, paths: set[str]) -> dict[str, np.ndarray]:
    """단일 패스로 필요한 프레임만 디코드 (gz tar는 random access 불가)."""
    from PIL import Image

    out: dict[str, np.ndarray] = {}
    with tarfile.open(data_root / "frames.tar.gz") as tf:
        for m in tf:
            if m.name in paths:
                out[m.name] = np.array(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
                if len(out) == len(paths):
                    break
    return out


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4점을 tl, tr, br, bl로 정렬."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # 반시계/시계 일관 순서
    # tl = x+y 최소가 첫 번째가 되도록 회전
    start = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start, axis=0)
    # tr이 두 번째가 되도록 방향 통일 (두 번째 점이 왼쪽이면 뒤집기)
    if pts[1, 0] < pts[3, 0]:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def mask_to_quad(mask: np.ndarray) -> tuple[np.ndarray | None, str]:
    """mask -> 4코너 quad. 실패 시 (None, 사유)."""
    import cv2
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return None, "empty_mask"
    largest = (lab == (np.bincount(lab.ravel())[1:].argmax() + 1)).astype(np.uint8)
    if largest.sum() < 0.05 * mask.size:
        return None, "too_small"
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
    quad_area = cv2.contourArea(quad)
    fill = float(largest.sum()) / max(quad_area, 1.0)
    if not (0.7 <= fill <= 1.3):
        return None, "quad_mask_mismatch"
    return order_corners(quad), "ok"


def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple) -> float:
    import cv2

    s = 4  # 다운스케일 래스터로 충분
    h, w = shape[0] // s, shape[1] // s
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [(a / s).astype(np.int32)], 1)
    cv2.fillPoly(mb, [(b / s).astype(np.int32)], 1)
    inter = float((ma & mb).sum())
    union = float((ma | mb).sum())
    return inter / max(union, 1.0)


def edge_rel_errors(det_quad: np.ndarray, gt_quad: np.ndarray, w_mm: float, h_mm: float) -> list[float]:
    """detected-quad homography로 GT 코너를 metric 평면에 사상해 변 길이 오차 측정."""
    import cv2

    dst = np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], np.float32)
    H = cv2.getPerspectiveTransform(det_quad, dst)
    gt_mm = cv2.perspectiveTransform(gt_quad.reshape(1, 4, 2), H).reshape(4, 2)
    tl, tr, br, bl = gt_mm
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    return [
        abs(top - w_mm) / w_mm,
        abs(bottom - w_mm) / w_mm,
        abs(left - h_mm) / h_mm,
        abs(right - h_mm) / h_mm,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/smartdoc"))
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--out-json", type=Path, default=OUT)
    args = ap.parse_args()

    from gaugeanything.segmenters import segment_sam3

    print("=== P2-1b SmartDoc detected-quad scale ===", flush=True)
    rows = load_metadata(args.data_root)
    sample = stratified_sample(rows, args.max_frames)
    print(f"frames: {len(sample)} (stratified over bg x document)", flush=True)
    images = extract_images(args.data_root, {r["image_path"] for r in sample})
    print(f"decoded {len(images)} frames", flush=True)

    per_prompt: dict[str, dict] = {}
    for prompt in PROMPTS:
        gate_fail: dict[str, int] = defaultdict(int)
        ious, corner_px, edge_errs = [], [], []
        for r in sample:
            img = images.get(r["image_path"])
            if img is None:
                gate_fail["missing_frame"] += 1
                continue
            gt = order_corners(np.array(
                [[float(r["tl_x"]), float(r["tl_y"])], [float(r["tr_x"]), float(r["tr_y"])],
                 [float(r["br_x"]), float(r["br_y"])], [float(r["bl_x"]), float(r["bl_y"])]],
                np.float32))
            w_mm = float(r.get("model_width") or 2100.0) * 0.1 or A4_W_MM
            h_mm = float(r.get("model_height") or 2970.0) * 0.1 or A4_H_MM
            try:
                instances = segment_sam3(img, prompt)
            except Exception as e:  # 모델/런타임 실패는 측정 불가로 집계
                gate_fail[f"sam3_error:{type(e).__name__}"] += 1
                continue
            if not instances:
                gate_fail["no_instance"] += 1
                continue
            best = max(instances, key=lambda i: i.score)
            quad, why = mask_to_quad(best.mask)
            if quad is None:
                gate_fail[why] += 1
                continue
            ious.append(quad_iou(quad, gt, img.shape[:2]))
            corner_px.append(float(np.linalg.norm(quad - gt, axis=1).mean()))
            edge_errs.extend(edge_rel_errors(quad, gt, w_mm, h_mm))
        n_ok = len(ious)
        e = np.array(edge_errs) if edge_errs else np.array([np.nan])
        per_prompt[prompt] = {
            "n_frames": len(sample),
            "n_measured": n_ok,
            "gate_pass_rate": round(n_ok / len(sample), 4),
            "gate_failures": dict(gate_fail),
            "quad_iou_median": round(float(np.median(ious)), 4) if ious else None,
            "corner_err_px_median": round(float(np.median(corner_px)), 2) if corner_px else None,
            "edge_rel_err_median": round(float(np.nanmedian(e)), 4),
            "edge_rel_err_p90": round(float(np.nanpercentile(e, 90)), 4),
        }
        print(f"  {prompt!r}: pass={per_prompt[prompt]['gate_pass_rate']:.2f} "
              f"iou={per_prompt[prompt]['quad_iou_median']} "
              f"edge_err={per_prompt[prompt]['edge_rel_err_median']}", flush=True)

    measured = {k: v for k, v in per_prompt.items() if v["n_measured"] > 0}
    best_prompt = min(measured, key=lambda k: measured[k]["edge_rel_err_median"]) if measured else None
    result = {
        "protocol": (
            "P2-1b: SAM3 promptable document mask -> quad gate -> detected-quad homography; "
            "GT corners mapped to the metric plane, edge lengths vs A4/model dims. "
            "GT-quad upper bound is 0% by construction (P2-1); this measures the detection cost. "
            "Gate failures are reported as not-measurable, never guessed."
        ),
        "dataset": "SmartDoc15-CH1",
        "n_frames": len(sample),
        "prompts": per_prompt,
        "best_prompt": best_prompt,
        "anchor_p2_1": {
            "gt_quad_upper_bound": 0.0,
            "naive_scale_median": "0.10-0.17 (smartdoc_scale_eval.json)",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
