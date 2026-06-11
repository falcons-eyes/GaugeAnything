"""Robustness audit for E-mm-3 krkCMd profile-width results.

The headline E-mm-3 number uses a deterministic Series/Image group split. This
audit asks whether that number is a lucky/cherry-picked split by evaluating:

  1. all 5 deterministic group folds
  2. leave-one-Series-out
  3. leave-one-Stage-out

For each split, the simple GaugeProfile-minrun5 predictor is optionally linearly
calibrated on the train side only, then evaluated on the held-out side. Author
DLM/AED table outputs are reported as fixed anchors.

Usage:
    python experiments/krkcmd_split_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.krkcmd_profile_eval import (  # noqa: E402
    linear_calibrate,
    read_rows,
    summarize,
    width_min_run,
)


def idx_where(rows: list[dict], pred) -> np.ndarray:
    return np.array([i for i, row in enumerate(rows) if pred(row)], dtype=int)


def evaluate_split(rows: list[dict], gt: np.ndarray, preds: dict[str, np.ndarray],
                   train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    calibrated, cal_meta = linear_calibrate(preds["GaugeProfile-minrun5"], gt, train_idx)
    return {
        "train_n": int(train_idx.size),
        "test_n": int(test_idx.size),
        "DLMwidth(author)": summarize(gt, preds["DLMwidth(author)"], test_idx),
        "AEDwidth(author)": summarize(gt, preds["AEDwidth(author)"], test_idx),
        "GaugeProfile-minrun5": summarize(gt, preds["GaugeProfile-minrun5"], test_idx),
        "GaugeProfile-minrun5+linear-cal": {
            **summarize(gt, calibrated, test_idx),
            "calibration": cal_meta,
        },
    }


def method_distribution(results: dict, method: str) -> dict:
    maes = np.array([v[method]["MAE_um"] for v in results.values()], dtype=float)
    return {
        "folds": int(maes.size),
        "MAE_mean_um": round(float(maes.mean()), 3),
        "MAE_std_um": round(float(maes.std(ddof=0)), 3),
        "MAE_min_um": round(float(maes.min()), 3),
        "MAE_max_um": round(float(maes.max()), 3),
    }


def main() -> int:
    rows = read_rows()
    gt = np.array([r["MANwidth"] for r in rows], dtype=np.float64)
    preds = {
        "DLMwidth(author)": np.array([r["DLMwidth"] for r in rows], dtype=np.float64),
        "AEDwidth(author)": np.array([r["AEDwidth"] for r in rows], dtype=np.float64),
        "GaugeProfile-minrun5": np.array([width_min_run(r["x"]) for r in rows], dtype=np.float64),
    }

    groups = sorted({r["group"] for r in rows})
    series = sorted({r["series"] for r in rows})
    stages = sorted({r["stage"] for r in rows})

    group_folds = {}
    for fold in range(5):
        test_groups = {g for i, g in enumerate(groups) if i % 5 == fold}
        test_idx = idx_where(rows, lambda r, tg=test_groups: r["group"] in tg)
        train_idx = idx_where(rows, lambda r, tg=test_groups: r["group"] not in tg)
        group_folds[f"group_fold_{fold}"] = evaluate_split(rows, gt, preds, train_idx, test_idx)

    series_folds = {}
    for held in series:
        test_idx = idx_where(rows, lambda r, h=held: r["series"] == h)
        train_idx = idx_where(rows, lambda r, h=held: r["series"] != h)
        series_folds[f"leave_series_{held}"] = evaluate_split(rows, gt, preds, train_idx, test_idx)

    stage_folds = {}
    for held in stages:
        test_idx = idx_where(rows, lambda r, h=held: r["stage"] == h)
        train_idx = idx_where(rows, lambda r, h=held: r["stage"] != h)
        stage_folds[f"leave_stage_{held}"] = evaluate_split(rows, gt, preds, train_idx, test_idx)

    methods = ["DLMwidth(author)", "AEDwidth(author)", "GaugeProfile-minrun5", "GaugeProfile-minrun5+linear-cal"]
    summary = {
        "group_5fold": {m: method_distribution(group_folds, m) for m in methods},
        "leave_one_series": {m: method_distribution(series_folds, m) for m in methods},
        "leave_one_stage": {m: method_distribution(stage_folds, m) for m in methods},
    }

    out = {
        "meta": {
            "dataset": "krkCMd",
            "n_profiles": len(rows),
            "n_groups": len(groups),
            "n_series": len(series),
            "n_stages": len(stages),
            "gt": "MANwidth",
            "note": "Audit for split sensitivity/cherry-picking risk. Calibration is fit on train split only.",
        },
        "summary": summary,
        "folds": {
            "group_5fold": group_folds,
            "leave_one_series": series_folds,
            "leave_one_stage": stage_folds,
        },
    }

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/krkcmd_split_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== krkCMd split robustness audit ===")
    print(json.dumps(out["meta"], indent=2, ensure_ascii=False))
    for split_name, split_summary in summary.items():
        print(f"\n[{split_name}] MAE mean ± std (min..max)")
        for method in methods:
            s = split_summary[method]
            print(
                f"  {method:<34} {s['MAE_mean_um']:>6.1f} ± {s['MAE_std_um']:<5.1f} "
                f"({s['MAE_min_um']:.1f}..{s['MAE_max_um']:.1f})"
            )
    print("\n결과 저장: experiments/results/krkcmd_split_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
