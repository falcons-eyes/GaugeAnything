"""E-dyn-3a: ADT ATEK projection audit.

This is the first ADT access sanity check after the gated download URL files
became available. It intentionally does not store or require signed URLs.

Protocol:
  ATEK cubercnn shard -> RGB frame + 3-D OBB GT + camera/device pose
  -> project 3-D boxes into the RGB camera -> compare with released 2-D boxes.

The goal is not to claim promptable segmentation yet. It verifies that the ADT
data path gives a metric, dynamic, egocentric geometry chain that can support
the next step: GT-box/mask upper bound, then SAM3 promptable object masks.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class AtekRecord:
    key: str
    sequence: str
    frame_id: int
    timestamp_ns: int
    image_bytes: bytes
    projection: np.ndarray
    t_device_camera: np.ndarray
    t_world_device: np.ndarray
    object_dimensions: np.ndarray
    t_world_object: np.ndarray
    instance_ids: np.ndarray
    boxes_xyxy: np.ndarray
    visibility: np.ndarray
    category_names: list[str]


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


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _load_torch(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    obj = torch.load(io.BytesIO(raw.read()), map_location="cpu")
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _load_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    return raw.read().decode("utf-8").strip()


def _load_json(tf: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    return json.loads(raw.read().decode("utf-8"))


def _load_bytes(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    raw = tf.extractfile(member)
    if raw is None:
        raise ValueError(f"Could not read {member.name}")
    return raw.read()


def iter_records(shards: Iterable[Path]) -> Iterable[AtekRecord]:
    for shard in shards:
        with tarfile.open(shard) as tf:
            groups: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
            for member in tf.getmembers():
                prefix, suffix = member.name.split(".", 1)
                groups[prefix][suffix] = member

            for prefix, files in sorted(groups.items()):
                gt = _load_json(tf, files["gt_data.json"])
                category_names = gt["obb3_gt"]["camera-rgb"]["category_names"]
                boxes_raw = _load_torch(tf, files["gt_data#obb2_gt+camera-rgb+box_ranges.pth"])
                boxes_xyxy = np.stack(
                    [boxes_raw[:, 0], boxes_raw[:, 2], boxes_raw[:, 1], boxes_raw[:, 3]],
                    axis=1,
                )
                yield AtekRecord(
                    key=prefix,
                    sequence=_load_text(tf, files["sequence_name.txt"]),
                    frame_id=int(_load_torch(tf, files["mfcd#camera-rgb+frame_ids.pth"])[0]),
                    timestamp_ns=int(_load_torch(tf, files["mtd#capture_timestamps_ns.pth"])[0]),
                    image_bytes=_load_bytes(tf, files["mfcd#camera-rgb+images_0.jpeg"]),
                    projection=_load_torch(tf, files["mfcd#camera-rgb+projection_params.pth"]),
                    t_device_camera=_load_torch(tf, files["mfcd#camera-rgb+t_device_camera.pth"]),
                    t_world_device=_load_torch(tf, files["mtd#ts_world_device.pth"]),
                    object_dimensions=_load_torch(tf, files["gt_data#obb3_gt+camera-rgb+object_dimensions.pth"]),
                    t_world_object=_load_torch(tf, files["gt_data#obb3_gt+camera-rgb+ts_world_object.pth"]),
                    instance_ids=_load_torch(tf, files["gt_data#obb3_gt+camera-rgb+instance_ids.pth"]),
                    boxes_xyxy=boxes_xyxy,
                    visibility=_load_torch(tf, files["gt_data#obb2_gt+camera-rgb+visibility_ratios.pth"]),
                    category_names=list(category_names),
                )


def project_box(record: AtekRecord, idx: int) -> np.ndarray | None:
    fx, fy, cx, cy = [float(x) for x in record.projection]
    t_camera_object = (
        np.linalg.inv(_homogeneous(record.t_device_camera))
        @ np.linalg.inv(_homogeneous(record.t_world_device))
        @ _homogeneous(record.t_world_object[idx])
    )
    pts = t_camera_object @ _box_corners(record.object_dimensions[idx])
    good = pts[2] > 1e-6
    if int(good.sum()) < 4:
        return None
    x = pts[0, good]
    y = pts[1, good]
    z = pts[2, good]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.asarray([u.min(), v.min(), u.max(), v.max()], dtype=float)


def _summarize_motion(records: list[AtekRecord]) -> dict:
    ordered = sorted(records, key=lambda r: r.timestamp_ns)
    speeds = []
    for a, b in zip(ordered, ordered[1:]):
        dt = (b.timestamp_ns - a.timestamp_ns) / 1e9
        if dt <= 0:
            continue
        pa = _homogeneous(a.t_world_device)[:3, 3]
        pb = _homogeneous(b.t_world_device)[:3, 3]
        speeds.append(float(np.linalg.norm(pb - pa) / dt))
    if not speeds:
        return {"n": 0}
    arr = np.asarray(speeds, dtype=float)
    return {
        "n": int(arr.size),
        "median_mps": float(np.median(arr)),
        "p90_mps": float(np.percentile(arr, 90)),
        "max_mps": float(arr.max()),
    }


def _write_overlay(records: list[AtekRecord], rows: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_frame = defaultdict(list)
    for row in sorted(rows, key=lambda x: x["iou"], reverse=True):
        by_frame[row["frame_id"]].append(row)

    selected = []
    used = set()
    for rec in records:
        if rec.frame_id in by_frame and rec.frame_id not in used:
            selected.append(rec)
            used.add(rec.frame_id)
        if len(selected) == 4:
            break
    if not selected:
        return

    thumbs = []
    for rec in selected:
        img = Image.open(io.BytesIO(rec.image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for row in by_frame[rec.frame_id][:8]:
            gt = row["gt_box_xyxy"]
            pr = row["projected_box_xyxy"]
            draw.rectangle(gt, outline=(0, 220, 120), width=5)
            draw.rectangle(pr, outline=(235, 80, 255), width=4)
            draw.text((gt[0], max(0, gt[1] - 18)), f"{row['category']} {row['iou']:.2f}", fill=(0, 220, 120))
        img.thumbnail((520, 520))
        thumbs.append(img)

    w = max(i.width for i in thumbs)
    h = max(i.height for i in thumbs)
    sheet = Image.new("RGB", (2 * w, 2 * h), (22, 24, 28))
    for k, img in enumerate(thumbs):
        sheet.paste(img, ((k % 2) * w, (k // 2) * h))
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        default=sorted(Path("datasets/adt_atek_sample").glob("*.tar")),
    )
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--min-box-px", type=float, default=8.0)
    ap.add_argument("--out-json", type=Path, default=Path("experiments/results/adt_atek_projection_audit.json"))
    ap.add_argument("--out-image", type=Path, default=Path("docs/assets/adt_atek_projection_audit.png"))
    args = ap.parse_args()

    records = list(iter_records(args.shards))
    rows = []
    categories = Counter()
    for rec in records:
        for i, (gt, vis, cat) in enumerate(zip(rec.boxes_xyxy, rec.visibility, rec.category_names)):
            if float(vis) < args.min_visibility:
                continue
            if (gt[2] - gt[0]) < args.min_box_px or (gt[3] - gt[1]) < args.min_box_px:
                continue
            pred = project_box(rec, i)
            if pred is None:
                continue
            iou = _box_iou(pred, gt)
            categories[cat] += 1
            rows.append(
                {
                    "sequence": rec.sequence,
                    "frame_id": rec.frame_id,
                    "category": cat,
                    "visibility": float(vis),
                    "dimensions_m": [float(x) for x in rec.object_dimensions[i]],
                    "gt_box_xyxy": [float(x) for x in gt],
                    "projected_box_xyxy": [float(x) for x in pred],
                    "iou": float(iou),
                    "center_err_px": float(np.linalg.norm(((pred[:2] + pred[2:]) - (gt[:2] + gt[2:])) / 2.0)),
                }
            )

    ious = np.asarray([r["iou"] for r in rows], dtype=float)
    center_err = np.asarray([r["center_err_px"] for r in rows], dtype=float)
    summary = {
        "dataset": "Aria Digital Twin ATEK cubercnn",
        "sequence": records[0].sequence if records else None,
        "shards": [p.name for p in args.shards],
        "n_frames": len(records),
        "n_instances_eval": int(ious.size),
        "filter": {"min_visibility": args.min_visibility, "min_box_px": args.min_box_px},
        "projection_chain": "inv(T_device_camera) @ inv(T_world_device) @ T_world_object, pinhole xyz",
        "mean_iou": float(ious.mean()) if ious.size else None,
        "median_iou": float(np.median(ious)) if ious.size else None,
        "p10_iou": float(np.percentile(ious, 10)) if ious.size else None,
        "pass_iou_0_50": float((ious >= 0.5).mean()) if ious.size else None,
        "pass_iou_0_75": float((ious >= 0.75).mean()) if ious.size else None,
        "median_center_err_px": float(np.median(center_err)) if center_err.size else None,
        "motion": _summarize_motion(records),
        "top_categories": categories.most_common(12),
    }
    result = {"summary": summary, "examples": sorted(rows, key=lambda r: r["iou"], reverse=True)[:40]}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_overlay(records, rows, args.out_image)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_image}")


if __name__ == "__main__":
    main()
