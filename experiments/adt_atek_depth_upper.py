"""E-dyn-3c: ADT EFM oracle-volume depth fusion upper bound.

This follows E-dyn-3b's negative box-only result. It asks a narrower question:

  If ADT gives us RGB-aligned depth and we use an oracle object volume/pose
  gate, can multiview depth recover metric object dimensions?

This is deliberately *not* a promptable SAM3 result. It is an upper-bound/data
sanity check for the next step: replace oracle volume gating with segmentation
or promptable masks.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EfmObservation:
    sequence: str
    instance_id: int
    category: str
    gt_dim_m: np.ndarray
    object_points: np.ndarray
    frame_key: str
    speed_mps: float


def _load_torch(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    obj = torch.load(io.BytesIO(raw.read()), map_location="cpu")
    return obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)


def _load_json(tf: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    return json.loads(raw.read().decode("utf-8"))


def _load_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    return raw.read().decode("utf-8").strip()


def _homogeneous(t: np.ndarray) -> np.ndarray:
    arr = np.asarray(t, dtype=float)
    if arr.shape == (1, 3, 4):
        arr = arr[0]
    out = np.eye(4, dtype=float)
    out[:3, :4] = arr
    return out


def _box_corners(dim: np.ndarray) -> np.ndarray:
    dx, dy, dz = np.asarray(dim, dtype=float) / 2.0
    pts = []
    for sx in (-dx, dx):
        for sy in (-dy, dy):
            for sz in (-dz, dz):
                pts.append((sx, sy, sz, 1.0))
    return np.asarray(pts, dtype=float).T


def _parse_fisheye624(params: np.ndarray) -> tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return fu, fv, cu, cv, k[6], p[2], s[4].

    ATEK stores the Project Aria Fisheye624 model in the compact 15-param form:
    [f, cu, cv, k0..k5, p0, p1, s0..s3]. Some tooling uses the 16-param
    [fu, fv, cu, cv, ...] form; support both.
    """
    p = np.asarray(params, dtype=float).reshape(-1)
    if len(p) == 15:
        f = float(p[0])
        return f, f, float(p[1]), float(p[2]), p[3:9], p[9:11], p[11:15]
    if len(p) == 16:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3]), p[4:10], p[10:12], p[12:16]
    raise ValueError(f"Expected 15 or 16 Fisheye624 params, got {len(p)}")


def _distort_ab(a: np.ndarray, b: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _fu, _fv, _cu, _cv, k, p, s = _parse_fisheye624(params)
    radius = np.sqrt(a * a + b * b)
    theta = np.arctan(radius)
    theta2 = theta * theta
    theta_distorted = theta.copy()
    theta_power = theta * theta2
    for coeff in k:
        theta_distorted = theta_distorted + coeff * theta_power
        theta_power = theta_power * theta2

    scale = np.ones_like(radius)
    nonzero = radius > 1e-12
    scale[nonzero] = theta_distorted[nonzero] / radius[nonzero]
    xr = a * scale
    yr = b * scale
    r2 = xr * xr + yr * yr

    p0, p1 = p
    s0, s1, s2, s3 = s
    tangential_x = 2 * p0 * xr * yr + p1 * (r2 + 2 * xr * xr)
    tangential_y = p0 * (r2 + 2 * yr * yr) + 2 * p1 * xr * yr
    thin_x = s0 * r2 + s1 * r2 * r2
    thin_y = s2 * r2 + s3 * r2 * r2
    return xr + tangential_x + thin_x, yr + tangential_y + thin_y


def fisheye624_project(points_camera: np.ndarray, params: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_camera, dtype=float)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    a = x / z
    b = y / z
    xd, yd = _distort_ab(a, b, params)
    fu, fv, cu, cv, _k, _p, _s = _parse_fisheye624(params)
    return np.stack([fu * xd + cu, fv * yd + cv], axis=-1)


def precompute_fisheye624_rays(params: np.ndarray, height: int = 240, width: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Numerically invert Fisheye624 for every pixel.

    Returns:
      z_rays: [a, b, 1] rays for z-depth interpretation.
      unit_rays: normalized rays for ray-distance interpretation.
    """
    fu, fv, cu, cv, _k, _p, _s = _parse_fisheye624(params)
    yy, xx = np.mgrid[0:height, 0:width]
    target_x = (xx.astype(float) - cu) / fu
    target_y = (yy.astype(float) - cv) / fv

    a = target_x.copy()
    b = target_y.copy()
    eps = 1e-5
    for _ in range(8):
        fx, fy = _distort_ab(a, b, params)
        rx = fx - target_x
        ry = fy - target_y
        fx_a, fy_a = _distort_ab(a + eps, b, params)
        fx_b, fy_b = _distort_ab(a, b + eps, params)
        j11 = (fx_a - fx) / eps
        j21 = (fy_a - fy) / eps
        j12 = (fx_b - fx) / eps
        j22 = (fy_b - fy) / eps
        det = j11 * j22 - j12 * j21
        good = np.abs(det) > 1e-9
        da = np.zeros_like(a)
        db = np.zeros_like(b)
        da[good] = (-rx[good] * j22[good] + j12[good] * ry[good]) / det[good]
        db[good] = (j21[good] * rx[good] - j11[good] * ry[good]) / det[good]
        a += da
        b += db
        if max(float(np.nanmax(np.abs(da))), float(np.nanmax(np.abs(db)))) < 1e-6:
            break

    z_rays = np.stack([a, b, np.ones_like(a)], axis=-1)
    unit_rays = z_rays / np.linalg.norm(z_rays, axis=-1, keepdims=True)
    return z_rays, unit_rays


def _iter_groups(shards: list[Path]):
    for shard in shards:
        with tarfile.open(shard) as tf:
            groups: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
            for member in tf.getmembers():
                prefix, suffix = member.name.split(".", 1)
                groups[prefix][suffix] = member
            for prefix, files in sorted(groups.items()):
                yield shard, tf, prefix, files


def _collect_observations(
    shards: list[Path],
    depth_mode: str,
    gate_mode: str,
    include_other: bool,
    volume_tolerance_m: float,
    min_bbox_px: int,
    min_points_per_frame: int,
) -> tuple[list[EfmObservation], dict]:
    observations: list[EfmObservation] = []
    ray_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    n_frames = 0
    n_candidates = 0
    n_projected = 0
    sequences = Counter()

    for shard, tf, prefix, files in _iter_groups(shards):
        sequence = _load_text(tf, files["sequence_name.txt"])
        gt_json = _load_json(tf, files["gt_data.json"])["efm_gt"]
        params = _load_torch(tf, files["mfcd#camera-rgb+projection_params.pth"])
        key = tuple(np.round(params.astype(float), 8).tolist())
        if key not in ray_cache:
            ray_cache[key] = precompute_fisheye624_rays(params)
        z_rays, unit_rays = ray_cache[key]
        base_rays = unit_rays if depth_mode == "ray" else z_rays

        depth = _load_torch(tf, files["mfcd#camera-rgb-depth+images.pth"])[:, 0]
        timestamps = _load_torch(tf, files["mfcd#camera-rgb+capture_timestamps_ns.pth"])
        t_world_device = _load_torch(tf, files["mtd#ts_world_device.pth"])
        t_device_camera = _load_torch(tf, files["mfcd#camera-rgb+t_device_camera.pth"])
        t_device_camera_inv = np.linalg.inv(_homogeneous(t_device_camera))
        positions = np.asarray([_homogeneous(t)[:3, 3] for t in t_world_device], dtype=float)
        frame_speeds = np.zeros(len(timestamps), dtype=float)
        if len(timestamps) > 1:
            for i in range(len(timestamps)):
                if i == 0:
                    j, k = 0, 1
                else:
                    j, k = i - 1, i
                dt = (float(timestamps[k]) - float(timestamps[j])) / 1e9
                frame_speeds[i] = float(np.linalg.norm(positions[k] - positions[j]) / dt) if dt > 0 else 0.0

        for frame_index, timestamp in enumerate(timestamps):
            ts_key = str(int(timestamp))
            if ts_key not in gt_json:
                continue
            n_frames += 1
            cam_gt = gt_json[ts_key]["camera-rgb"]
            names = cam_gt["category_names"]
            base = f"gt_data#efm_gt+{ts_key}+camera-rgb+"
            dims = _load_torch(tf, files[base + "object_dimensions.pth"])
            instance_ids = _load_torch(tf, files[base + "instance_ids.pth"])
            t_world_object = _load_torch(tf, files[base + "ts_world_object.pth"])
            t_camera_world = t_device_camera_inv @ np.linalg.inv(_homogeneous(t_world_device[frame_index]))
            valid_depth = depth[frame_index] > 0

            for object_index, (dim, instance_id, name) in enumerate(zip(dims, instance_ids, names)):
                if name == "other" and not include_other:
                    continue
                n_candidates += 1
                t_camera_object = t_camera_world @ _homogeneous(t_world_object[object_index])
                corners_camera = (t_camera_object @ _box_corners(dim))[:3].T
                if int((corners_camera[:, 2] > 0).sum()) < 4:
                    continue
                uv = fisheye624_project(corners_camera, params)
                if not np.all(np.isfinite(uv)):
                    continue
                x0, y0 = np.floor(uv.min(axis=0) - 2).astype(int)
                x1, y1 = np.ceil(uv.max(axis=0) + 2).astype(int)
                if x1 < 0 or y1 < 0 or x0 >= 240 or y0 >= 240:
                    continue
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(239, x1)
                y1 = min(239, y1)
                if (x1 - x0) < min_bbox_px or (y1 - y0) < min_bbox_px:
                    continue
                n_projected += 1

                crop = np.s_[y0 : y1 + 1, x0 : x1 + 1]
                valid = valid_depth[crop]
                if int(valid.sum()) < min_points_per_frame:
                    continue

                points_camera = base_rays[crop] * depth[frame_index][crop][..., None]
                points_camera = points_camera[valid]
                ph = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1).T
                points_object = (np.linalg.inv(t_camera_object) @ ph)[:3].T
                if gate_mode == "volume":
                    inside = np.all(
                        np.abs(points_object) <= (np.asarray(dim, dtype=float) / 2 + volume_tolerance_m),
                        axis=1,
                    )
                elif gate_mode == "roi":
                    inside = np.ones(points_object.shape[0], dtype=bool)
                else:
                    raise ValueError(f"Unknown gate_mode: {gate_mode}")
                if int(inside.sum()) < min_points_per_frame:
                    continue
                observations.append(
                    EfmObservation(
                        sequence=sequence,
                        instance_id=int(instance_id),
                        category=str(name),
                        gt_dim_m=np.asarray(dim, dtype=float),
                        object_points=points_object[inside],
                        frame_key=f"{shard.name}:{prefix}:{frame_index}",
                        speed_mps=float(frame_speeds[frame_index]),
                    )
                )
                sequences[sequence] += 1

    meta = {
        "n_frames": n_frames,
        "n_object_candidates": n_candidates,
        "n_projected_candidates": n_projected,
        "n_frame_observations": len(observations),
        "sequences": sorted(sequences),
    }
    return observations, meta


def summarize_observations(
    observations: list[EfmObservation],
    min_views: int,
    min_points: int,
    low_percentile: float,
    high_percentile: float,
) -> dict:
    grouped: dict[tuple[str, int], list[EfmObservation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.sequence, obs.instance_id)].append(obs)

    rows = []
    for (sequence, instance_id), items in sorted(grouped.items()):
        frame_keys = {item.frame_key for item in items}
        points = np.concatenate([item.object_points for item in items], axis=0)
        if len(frame_keys) < min_views or points.shape[0] < min_points:
            continue
        lo = np.percentile(points, low_percentile, axis=0)
        hi = np.percentile(points, high_percentile, axis=0)
        est_dim = hi - lo
        gt_dim = np.median([item.gt_dim_m for item in items], axis=0)
        rel = np.abs(est_dim - gt_dim) / np.maximum(gt_dim, 1e-6)
        speed = np.asarray([item.speed_mps for item in items], dtype=float)
        rows.append(
            {
                "sequence": sequence,
                "instance_id": int(instance_id),
                "category": items[0].category,
                "n_views": int(len(frame_keys)),
                "n_points": int(points.shape[0]),
                "speed_mps_median": float(np.median(speed)),
                "speed_mps_p90": float(np.percentile(speed, 90)),
                "pred_dim_m": [float(x) for x in est_dim],
                "gt_dim_m": [float(x) for x in gt_dim],
                "axis_rel_errors": [float(x) for x in rel],
                "axis_median_rel_error": float(np.median(rel)),
                "axis_mean_rel_error": float(rel.mean()),
            }
        )

    med = np.asarray([row["axis_median_rel_error"] for row in rows], dtype=float)
    mean = np.asarray([row["axis_mean_rel_error"] for row in rows], dtype=float)
    cats = Counter(row["category"] for row in rows)

    def summarize_subset(subset: list[dict]) -> dict:
        vals = np.asarray([row["axis_median_rel_error"] for row in subset], dtype=float)
        speeds = np.asarray([row["speed_mps_median"] for row in subset], dtype=float)
        return {
            "n": int(len(subset)),
            "axis_median_rel_error_median": float(np.median(vals)) if vals.size else None,
            "axis_median_rel_error_p90": float(np.percentile(vals, 90)) if vals.size else None,
            "speed_mps_median": float(np.median(speeds)) if speeds.size else None,
        }

    by_sequence = {
        sequence: summarize_subset([row for row in rows if row["sequence"] == sequence])
        for sequence in sorted({row["sequence"] for row in rows})
    }
    speed_bins = [
        ("0.00-0.10", 0.0, 0.10),
        ("0.10-0.25", 0.10, 0.25),
        ("0.25-0.50", 0.25, 0.50),
        ("0.50+", 0.50, float("inf")),
    ]
    by_speed_bin = {
        label: summarize_subset([row for row in rows if lo <= row["speed_mps_median"] < hi])
        for label, lo, hi in speed_bins
    }
    return {
        "rows": rows,
        "summary": {
            "n_instances_fit": int(len(rows)),
            "axis_median_rel_error_median": float(np.median(med)) if med.size else None,
            "axis_median_rel_error_p90": float(np.percentile(med, 90)) if med.size else None,
            "axis_mean_rel_error_median": float(np.median(mean)) if mean.size else None,
            "pass_axis_median_rel_10pct": float((med <= 0.10).mean()) if med.size else None,
            "pass_axis_median_rel_25pct": float((med <= 0.25).mean()) if med.size else None,
            "top_categories": cats.most_common(16),
            "by_sequence": by_sequence,
            "by_speed_bin": by_speed_bin,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        default=sorted(Path("datasets/adt_atek_efm").glob("*.tar")),
    )
    ap.add_argument("--depth-mode", choices=["ray", "z"], default="ray")
    ap.add_argument(
        "--gate-mode",
        choices=["volume", "roi"],
        default="volume",
        help="volume = GT object volume oracle; roi = projected 3-D box ROI only, no object-volume filtering.",
    )
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--volume-tolerance-m", type=float, default=0.02)
    ap.add_argument("--min-bbox-px", type=int, default=3)
    ap.add_argument("--min-points-per-frame", type=int, default=5)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--low-percentile", type=float, default=2.0)
    ap.add_argument("--high-percentile", type=float, default=98.0)
    ap.add_argument("--out-json", type=Path, default=Path("experiments/results/adt_atek_depth_upper.json"))
    args = ap.parse_args()

    observations, meta = _collect_observations(
        shards=args.shards,
        depth_mode=args.depth_mode,
        gate_mode=args.gate_mode,
        include_other=args.include_other,
        volume_tolerance_m=args.volume_tolerance_m,
        min_bbox_px=args.min_bbox_px,
        min_points_per_frame=args.min_points_per_frame,
    )
    summarized = summarize_observations(
        observations=observations,
        min_views=args.min_views,
        min_points=args.min_points,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    rows = summarized["rows"]
    summary = {
        "dataset": "Aria Digital Twin ATEK efm",
        "sequences": meta["sequences"],
        "shards": [p.name for p in args.shards],
        "protocol": (
            f"{args.gate_mode} depth fusion: Fisheye624 RGB-depth rays + GT object/world/device/camera poses; "
            "points are fused in object coordinates and robust 2-98 percentile extents are compared to GT dimensions."
        ),
        "depth_mode": args.depth_mode,
        "gate_mode": args.gate_mode,
        "filter": {
            "include_other": args.include_other,
            "volume_tolerance_m": args.volume_tolerance_m,
            "min_bbox_px": args.min_bbox_px,
            "min_points_per_frame": args.min_points_per_frame,
            "min_views": args.min_views,
            "min_points": args.min_points,
            "extent_percentiles": [args.low_percentile, args.high_percentile],
        },
        **meta,
        **summarized["summary"],
        "note": (
            "volume mode is an oracle-volume upper bound, not a promptable SAM3 result. "
            "roi mode is a negative control without object-volume filtering."
        ),
    }
    result = {
        "summary": summary,
        "best_examples": sorted(rows, key=lambda r: r["axis_median_rel_error"])[:24],
        "worst_examples": sorted(rows, key=lambda r: r["axis_median_rel_error"], reverse=True)[:24],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
