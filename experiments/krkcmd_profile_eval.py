"""E-mm-3 — krkCMd crack-width physical-GT evaluation.

krkCMd exposes 19,098 scanner brightness profiles with manual crack width
(`MANwidth`, micrometers), plus the authors' AED and DLM outputs. The full image
zip is large (~38 GB), but this table is enough for a physical-unit crack-width
benchmark over 501-pixel cross-crack profiles.

This script reports:
  - published table baselines: AEDwidth, DLMwidth
  - deterministic GaugeProfile rules over the 1-D profile
  - optional linear calibration learned only on a group split

Usage:
    python experiments/krkcmd_profile_eval.py --gallery docs/assets
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("datasets/krkcmd/krkCMd_table.csv")
PX_TO_UM = 1000.0 * 25.4 / 6400.0  # scanner resolution documented by krkCMd scripts
N_PROFILE = 501


def read_rows(path: Path = ROOT) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = np.array([float(row[f"x{i}"]) for i in range(1, N_PROFILE + 1)], dtype=np.float32)
            rows.append(
                {
                    "no": int(row["No"]),
                    "profile": row["Profile"],
                    "series": row["Series"],
                    "image": row["Image"],
                    "stage": int(row["Stage"]),
                    "group": f"{row['Series']}/Image{row['Image']}",
                    "x": x,
                    "MANwidth": float(row["MANwidth"]),
                    "DLMwidth": float(row["DLMwidth"]),
                    "AEDwidth": float(row["AEDwidth"]),
                }
            )
    return rows


def contiguous_width(mask: np.ndarray, center: int) -> int:
    if not mask[center]:
        return 0
    left = center
    while left > 0 and mask[left - 1]:
        left -= 1
    right = center
    while right + 1 < mask.size and mask[right + 1]:
        right += 1
    return int(right - left + 1)


def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    sum_total = float((hist * centers).sum())
    weight_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    best_thr = float(centers[0])
    for count, center in zip(hist, centers):
        weight_b += float(count)
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += float(count) * float(center)
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = float(center)
    return best_thr


def width_min_run(profile: np.ndarray, accuracy: float = 5.0) -> float:
    center = int(np.argmin(profile))
    mask = profile <= float(profile[center] + accuracy)
    return contiguous_width(mask, center) * PX_TO_UM


def width_otsu_component(profile: np.ndarray) -> float:
    center = int(np.argmin(profile))
    thr = otsu_threshold(profile)
    mask = profile <= thr
    return contiguous_width(mask, center) * PX_TO_UM


def width_half_depth(profile: np.ndarray, window: int = 80) -> float:
    center = int(np.argmin(profile))
    lo = max(0, center - window)
    hi = min(profile.size, center + window + 1)
    left_peak = float(profile[lo : center + 1].max()) if center > lo else float(profile[center])
    right_peak = float(profile[center:hi].max()) if center + 1 < hi else float(profile[center])
    edge = 0.5 * (left_peak + right_peak)
    valley = float(profile[center])
    thr = valley + 0.5 * max(edge - valley, 0.0)
    mask = profile <= thr
    return contiguous_width(mask, center) * PX_TO_UM


def group_split(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({row["group"] for row in rows})
    test_groups = {g for i, g in enumerate(groups) if i % 5 == 0}
    train, test = [], []
    for i, row in enumerate(rows):
        (test if row["group"] in test_groups else train).append(i)
    return np.array(train, dtype=int), np.array(test, dtype=int)


def linear_calibrate(pred: np.ndarray, gt: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    x = pred[train_idx]
    y = gt[train_idx]
    A = np.stack([x, np.ones_like(x)], axis=1)
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    out = np.maximum(0.0, slope * pred + intercept)
    return out, {"slope": round(float(slope), 6), "intercept": round(float(intercept), 3)}


def summarize(gt: np.ndarray, pred: np.ndarray, idx: np.ndarray | None = None) -> dict:
    if idx is not None:
        gt = gt[idx]
        pred = pred[idx]
    err = pred - gt
    abs_err = np.abs(err)
    denom = np.maximum(gt, PX_TO_UM)
    rel = abs_err / denom
    if np.std(gt) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(gt, pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": int(gt.size),
        "MAE_um": round(float(abs_err.mean()), 3),
        "RMSE_um": round(float(np.sqrt(np.mean(err**2))), 3),
        "median_abs_err_um": round(float(np.median(abs_err)), 3),
        "bias_um": round(float(err.mean()), 3),
        "rel_err_median": round(float(np.median(rel)), 3),
        "pass@50um": round(float((abs_err <= 50.0).mean()), 3),
        "pass@100um": round(float((abs_err <= 100.0).mean()), 3),
        "pearson_r": round(corr, 4) if math.isfinite(corr) else None,
    }


def make_gallery(rows: list[dict], preds: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.array([r["MANwidth"] for r in rows])
    half = preds["GaugeProfile-halfdepth"]
    err = np.abs(half - gt)
    picks = [
        int(np.argmin(np.abs(gt - 50))),
        int(np.argmin(np.abs(gt - 120))),
        int(np.argmin(np.abs(gt - 300))),
        int(np.argmax(err)),
    ]
    labels = ["thin", "median-ish", "wide", "worst half-depth"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, idx, label in zip(axes.ravel(), picks, labels):
        row = rows[idx]
        x = row["x"]
        px = np.arange(1, N_PROFILE + 1)
        ax.plot(px, x, lw=1.5, color="#1b4d89")
        ax.axvline(int(np.argmin(x)) + 1, color="#d1495b", lw=1, alpha=0.8)
        ax.set_title(
            f"{label}: GT {row['MANwidth']:.1f}um | half {half[idx]:.1f}um | "
            f"AED {row['AEDwidth']:.1f}um | DLM {row['DLMwidth']:.1f}um"
        )
        ax.set_xlabel("profile pixel")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.25)
    fig.suptitle("E-mm-3 krkCMd: 501-pixel cross-crack brightness profiles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT))
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()

    table_path = Path(args.table)
    if not table_path.exists():
        print(f"missing {table_path}; run: bash data/scripts/download_metric.sh krkcmd")
        return 1

    rows = read_rows(table_path)
    gt = np.array([r["MANwidth"] for r in rows], dtype=np.float64)
    train_idx, test_idx = group_split(rows)

    preds = {
        "DLMwidth(author)": np.array([r["DLMwidth"] for r in rows], dtype=np.float64),
        "AEDwidth(author)": np.array([r["AEDwidth"] for r in rows], dtype=np.float64),
        "GaugeProfile-minrun5": np.array([width_min_run(r["x"]) for r in rows], dtype=np.float64),
        "GaugeProfile-otsu": np.array([width_otsu_component(r["x"]) for r in rows], dtype=np.float64),
        "GaugeProfile-halfdepth": np.array([width_half_depth(r["x"]) for r in rows], dtype=np.float64),
    }

    cal_meta = {}
    for name in ["GaugeProfile-minrun5", "GaugeProfile-otsu", "GaugeProfile-halfdepth"]:
        cal, meta = linear_calibrate(preds[name], gt, train_idx)
        preds[f"{name}+linear-cal"] = cal
        cal_meta[f"{name}+linear-cal"] = meta

    summary = {}
    for name, pred in preds.items():
        summary[name] = {
            "all": summarize(gt, pred),
            "group_split_train": summarize(gt, pred, train_idx),
            "group_split_test": summarize(gt, pred, test_idx),
        }
        if name in cal_meta:
            summary[name]["calibration"] = cal_meta[name]

    meta = {
        "dataset": "krkCMd",
        "n_profiles": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "train_profiles": int(train_idx.size),
        "test_profiles": int(test_idx.size),
        "unit": "um",
        "px_to_um": PX_TO_UM,
        "gt": "MANwidth",
        "note": "profile-level physical-width benchmark; full image zip not required",
    }

    print("=== E-mm-3 krkCMd profile-width evaluation ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n{'method':<34}{'test MAE':>10}{'test RMSE':>11}{'test medAE':>12}{'pass@50':>10}{'r':>8}")
    print("-" * 86)
    for name, s in sorted(summary.items(), key=lambda kv: kv[1]["group_split_test"]["MAE_um"]):
        t = s["group_split_test"]
        print(
            f"{name:<34}{t['MAE_um']:>10.1f}{t['RMSE_um']:>11.1f}"
            f"{t['median_abs_err_um']:>12.1f}{t['pass@50um']*100:>9.1f}%{t['pearson_r']:>8}"
        )

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "krkcmd_profile_eval.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.gallery:
        make_gallery(rows, preds, Path(args.gallery) / "krkcmd_profile.png")
    print("\n결과 저장: experiments/results/krkcmd_profile_eval.json")
    if args.gallery:
        print("갤러리 저장:", Path(args.gallery) / "krkcmd_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
