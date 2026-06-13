"""E-dyn-3b: ADT box-only multiview dimension upper-bound smoke.

This intentionally uses strong GT help: released 2-D boxes plus ADT object pose,
camera pose, and intrinsics. It estimates a constant 3-D object dimension per
instance from multiple frames, then compares to ADT object_dimensions.

Interpretation: if this box-only upper bound is weak, SAM3 box/mask results
should not be overclaimed without depth, masks, or stronger multiview fusion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from adt_atek_projection_audit import _box_corners, _homogeneous, iter_records


def project_dim(record, idx: int, dim: np.ndarray) -> np.ndarray | None:
    fx, fy, cx, cy = [float(x) for x in record.projection]
    t_camera_object = (
        np.linalg.inv(_homogeneous(record.t_device_camera))
        @ np.linalg.inv(_homogeneous(record.t_world_device))
        @ _homogeneous(record.t_world_object[idx])
    )
    pts = t_camera_object @ _box_corners(dim)
    good = pts[2] > 1e-6
    if int(good.sum()) < 4:
        return None
    u = fx * pts[0, good] / pts[2, good] + cx
    v = fy * pts[1, good] / pts[2, good] + cy
    return np.asarray([u.min(), v.min(), u.max(), v.max()], dtype=float)


def residual(log_dim: np.ndarray, obs: list[dict]) -> np.ndarray:
    dim = np.exp(log_dim)
    out = []
    for item in obs:
        pred = project_dim(item["record"], item["idx"], dim)
        if pred is None:
            continue
        gt = item["box"]
        scale = max(20.0, float(gt[2] - gt[0]), float(gt[3] - gt[1]))
        out.extend(((pred - gt) / scale).tolist())
    return np.asarray(out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        default=sorted(Path("datasets/adt_atek_sample").glob("*.tar")),
    )
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--min-box-px", type=float, default=16.0)
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=Path("experiments/results/adt_atek_box_dimension_upper.json"))
    args = ap.parse_args()

    records = list(iter_records(args.shards))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        for idx, (inst, box, vis, cat) in enumerate(zip(rec.instance_ids, rec.boxes_xyxy, rec.visibility, rec.category_names)):
            if float(vis) < args.min_visibility:
                continue
            if (box[2] - box[0]) < args.min_box_px or (box[3] - box[1]) < args.min_box_px:
                continue
            grouped[int(inst)].append(
                {
                    "record": rec,
                    "idx": idx,
                    "box": np.asarray(box, dtype=float),
                    "category": cat,
                    "gt_dim": np.asarray(rec.object_dimensions[idx], dtype=float),
                    "frame_id": rec.frame_id,
                }
            )

    starts = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    rows = []
    for inst, obs in sorted(grouped.items()):
        if len(obs) < args.min_views:
            continue
        best = None
        for start in starts:
            fit = least_squares(
                residual,
                np.log([start, start, start]),
                args=(obs,),
                bounds=(np.log([0.01, 0.01, 0.005]), np.log([5.0, 5.0, 5.0])),
                max_nfev=250,
            )
            if best is None or fit.cost < best.cost:
                best = fit
        assert best is not None
        pred = np.exp(best.x)
        gt = np.median([o["gt_dim"] for o in obs], axis=0)
        rel = np.abs(pred - gt) / np.maximum(gt, 1e-6)
        rows.append(
            {
                "instance_id": inst,
                "category": obs[0]["category"],
                "n_views": len(obs),
                "pred_dim_m": [float(x) for x in pred],
                "gt_dim_m": [float(x) for x in gt],
                "axis_rel_errors": [float(x) for x in rel],
                "axis_median_rel_error": float(np.median(rel)),
                "axis_mean_rel_error": float(rel.mean()),
                "fit_cost": float(best.cost),
            }
        )

    med = np.asarray([r["axis_median_rel_error"] for r in rows], dtype=float)
    mean = np.asarray([r["axis_mean_rel_error"] for r in rows], dtype=float)
    cats = Counter(r["category"] for r in rows)
    summary = {
        "dataset": "Aria Digital Twin ATEK cubercnn",
        "sequence": records[0].sequence if records else None,
        "shards": [p.name for p in args.shards],
        "n_frames": len(records),
        "n_instances_fit": int(len(rows)),
        "filter": {
            "min_visibility": args.min_visibility,
            "min_box_px": args.min_box_px,
            "min_views": args.min_views,
        },
        "protocol": "Fit one constant 3-D dimension vector per instance from released 2-D boxes, GT poses, and intrinsics.",
        "axis_median_rel_error_median": float(np.median(med)) if med.size else None,
        "axis_median_rel_error_p90": float(np.percentile(med, 90)) if med.size else None,
        "axis_mean_rel_error_median": float(np.median(mean)) if mean.size else None,
        "pass_axis_median_rel_10pct": float((med <= 0.10).mean()) if med.size else None,
        "pass_axis_median_rel_25pct": float((med <= 0.25).mean()) if med.size else None,
        "top_categories": cats.most_common(12),
        "note": "Box-only inverse dimension is ill-posed for many object poses; depth/mask/multiview constraints are needed before promptable ADT claims.",
    }
    result = {
        "summary": summary,
        "best_examples": sorted(rows, key=lambda r: r["axis_median_rel_error"])[:20],
        "worst_examples": sorted(rows, key=lambda r: r["axis_median_rel_error"], reverse=True)[:20],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
