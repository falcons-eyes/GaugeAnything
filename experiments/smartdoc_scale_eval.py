"""P2-1 SmartDoc known-document scale adapter.

This is a marker-free physical-scale stress test:
  - Data: SmartDoc15-CH1 frames metadata (A4 documents, GT quadrilateral)
  - Naive baseline: one global mm/px from the apparent document width/height
  - Plane upper-bound: GT quadrilateral defines the plane homography, so A4
    dimensions are recovered by construction; the useful number here is how
    badly naive scale fails under realistic handheld perspective.

The script intentionally needs only metadata.csv.gz from frames.tar.gz; images
are not required for this smoke metric.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def _open_metadata(data_root: Path):
    candidates = [
        data_root / "metadata.csv.gz",
        data_root / "frames" / "metadata.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return gzip.open(path, "rt", encoding="utf-8", newline="")

    archive = data_root / "frames.tar.gz"
    if archive.exists():
        tf = tarfile.open(archive)
        member = next((m for m in tf.getmembers() if m.name.endswith("metadata.csv.gz")), None)
        if member is None:
            tf.close()
            raise FileNotFoundError(f"metadata.csv.gz not found inside {archive}")
        raw = tf.extractfile(member)
        if raw is None:
            tf.close()
            raise FileNotFoundError(f"Could not read {member.name} from {archive}")
        # Keep tarfile alive by materializing the small compressed metadata.
        payload = raw.read()
        tf.close()
        return gzip.open(io.BytesIO(payload), "rt", encoding="utf-8", newline="")

    raise FileNotFoundError(
        f"SmartDoc metadata not found under {data_root}. Run: DATA_ROOT=./datasets bash data/scripts/download_metric.sh smartdoc"
    )


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def row_metrics(row: dict) -> dict:
    tl = np.array([float(row["tl_x"]), float(row["tl_y"])])
    bl = np.array([float(row["bl_x"]), float(row["bl_y"])])
    br = np.array([float(row["br_x"]), float(row["br_y"])])
    tr = np.array([float(row["tr_x"]), float(row["tr_y"])])

    top = dist(tl, tr)
    bottom = dist(bl, br)
    left = dist(tl, bl)
    right = dist(tr, br)
    width_px = (top + bottom) / 2.0
    height_px = (left + right) / 2.0
    if width_px <= 1e-6 or height_px <= 1e-6:
        raise ValueError("degenerate quad")

    model_w_mm = float(row.get("model_width", 2100.0)) * 0.1
    model_h_mm = float(row.get("model_height", 2970.0)) * 0.1
    if not math.isfinite(model_w_mm) or model_w_mm <= 0:
        model_w_mm = A4_WIDTH_MM
    if not math.isfinite(model_h_mm) or model_h_mm <= 0:
        model_h_mm = A4_HEIGHT_MM

    naive_scale_from_width = model_w_mm / width_px
    naive_scale_from_height = model_h_mm / height_px
    height_from_width_scale = height_px * naive_scale_from_width
    width_from_height_scale = width_px * naive_scale_from_height

    return {
        "width_px": width_px,
        "height_px": height_px,
        "model_w_mm": model_w_mm,
        "model_h_mm": model_h_mm,
        "naive_height_rel_error": abs(height_from_width_scale - model_h_mm) / model_h_mm,
        "naive_width_rel_error": abs(width_from_height_scale - model_w_mm) / model_w_mm,
        "top_bottom_ratio": max(top, bottom) / max(min(top, bottom), 1e-6),
        "left_right_ratio": max(left, right) / max(min(left, right), 1e-6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/smartdoc"))
    ap.add_argument("--max-frames", type=int, default=5000)
    ap.add_argument("--out-json", type=Path, default=Path("experiments/results/smartdoc_scale_eval.json"))
    args = ap.parse_args()

    rows = []
    with _open_metadata(args.data_root) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(row_metrics(row))
            except Exception:
                continue
            if len(rows) >= args.max_frames:
                break

    if not rows:
        raise RuntimeError("No valid SmartDoc metadata rows parsed")

    h = np.array([r["naive_height_rel_error"] for r in rows], dtype=float)
    w = np.array([r["naive_width_rel_error"] for r in rows], dtype=float)
    persp = np.maximum(
        np.array([r["top_bottom_ratio"] for r in rows], dtype=float),
        np.array([r["left_right_ratio"] for r in rows], dtype=float),
    )
    summary = {
        "dataset": "SmartDoc15-CH1",
        "protocol": "Known A4 quadrilateral scale smoke test. Naive global mm/px is compared against GT A4 dimensions; GT homography is a 0-error upper bound by construction.",
        "n_frames": int(len(rows)),
        "a4_width_mm": A4_WIDTH_MM,
        "a4_height_mm": A4_HEIGHT_MM,
        "naive_height_rel_error_median": float(np.median(h)),
        "naive_height_rel_error_p90": float(np.percentile(h, 90)),
        "naive_width_rel_error_median": float(np.median(w)),
        "naive_width_rel_error_p90": float(np.percentile(w, 90)),
        "perspective_ratio_median": float(np.median(persp)),
        "perspective_ratio_p90": float(np.percentile(persp, 90)),
        "plane_homography_upper_bound_error": 0.0,
        "note": "This uses GT quadrilaterals, not a detected document mask. Next step: replace GT quad with detector/SAM3/document prompt.",
    }
    result = {"summary": summary, "sample_rows": rows[:24]}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
