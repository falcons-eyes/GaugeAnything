"""M2 v2-a — tiny measurement specialist over SAM-mask/image features.

Purpose
-------
The first neural M2 refiner improved raw SAM3 width error, but a 5-bin
quantile calibration beat it. This script sets the next honest bar for a
GaugeAnything-owned model:

  - train only on train/val source groups from the existing M2 cache
  - use no source id and no held-out test labels for model selection
  - compare raw width, quantile calibration, the old neural M2, and small
    tabular specialists that predict the measured width directly

This is not meant to be the final model. It is the smallest useful
"measurement head" experiment before investing in DINO/SAM/Depth encoder
fine-tuning.

Usage on Spark:
    .venv/bin/python experiments/m2_specialist_tabular.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.m2_refiner import CACHE, width_of  # noqa: E402


OUT = Path("experiments/results/m2_specialist_tabular.json")
CKPT = Path("checkpoints/gaugehead_tiny_width.pkl")
FEATURE_CACHE = CACHE / "tabular_features_v1.npz"


@dataclass
class SplitData:
    x: np.ndarray
    y_width: np.ndarray
    raw_width: np.ndarray
    src: np.ndarray
    feature_names: list[str]


def _bbox_features(mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {
            "area": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_fill": 0.0,
            "components": 0.0,
            "largest_frac": 0.0,
            "skel_len": 0.0,
            "skel_width": 0.0,
        }
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    bbox_w = float(xs.max() - xs.min() + 1)
    bbox_h = float(ys.max() - ys.min() + 1)
    bbox_area = max(bbox_w * bbox_h, 1.0)
    lab, n_comp = ndimage.label(mask)
    counts = np.bincount(lab.ravel())[1:]
    largest = float(counts.max()) if len(counts) else 0.0
    # Fast proxy for centerline length. Full skeletonization is overkill for
    # this tabular screening run and made iteration painfully slow on Spark.
    col_coverage = float(np.count_nonzero(mask.any(axis=0)))
    row_coverage = float(np.count_nonzero(mask.any(axis=1)))
    skel_len = max(col_coverage, row_coverage)
    return {
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_fill": area / bbox_area,
        "components": float(n_comp),
        "largest_frac": largest / max(area, 1.0),
        "skel_len": skel_len,
        "skel_width": area / max(skel_len, 1.0),
    }


def _image_features(img_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    img = img_u8.astype(np.float32) / 255.0
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)
    vals = img[mask]
    bg = img[~mask]
    gvals = grad[mask]
    if len(vals) == 0:
        vals = np.array([0.0], dtype=np.float32)
    if len(bg) == 0:
        bg = np.array([0.0], dtype=np.float32)
    if len(gvals) == 0:
        gvals = np.array([0.0], dtype=np.float32)
    return {
        "fg_mean": float(vals.mean()),
        "fg_std": float(vals.std()),
        "bg_mean": float(bg.mean()),
        "contrast": float(bg.mean() - vals.mean()),
        "grad_mask_mean": float(gvals.mean()),
        "grad_global_mean": float(grad.mean()),
    }


def row_features(img_u8: np.ndarray, sam: np.ndarray) -> tuple[list[float], list[str]]:
    sam = sam.astype(bool)
    raw_w = float(width_of(sam)) if sam.sum() >= 20 else 0.0
    feats = {"raw_width": raw_w, "raw_width_log": float(np.log1p(raw_w))}
    feats.update(_bbox_features(sam))
    feats.update(_image_features(img_u8, sam))
    # Scale-stable ratios for cross-source transfer.
    feats["area_sqrt"] = float(np.sqrt(max(feats["area"], 0.0)))
    feats["width_over_bbox_h"] = raw_w / max(feats["bbox_h"], 1.0)
    feats["width_over_skel_width"] = raw_w / max(feats["skel_width"], 1.0)
    names = sorted(feats)
    return [float(feats[n]) for n in names], names


def load_split(split: str) -> SplitData:
    d = np.load(CACHE / f"{split}.npz", allow_pickle=True)
    rows: list[list[float]] = []
    y: list[float] = []
    raw: list[float] = []
    src: list[str] = []
    names: list[str] | None = None
    for i in range(len(d["imgs"])):
        gt = d["gts"][i].astype(bool)
        w_gt = float(width_of(gt)) if gt.sum() >= 20 else 0.0
        if w_gt <= 0:
            continue
        f, n = row_features(d["imgs"][i], d["sams"][i].astype(bool))
        if names is None:
            names = n
        rows.append(f)
        y.append(w_gt)
        raw.append(f[n.index("raw_width")])
        src.append(str(d["srcs"][i]))
        if (i + 1) % 500 == 0:
            print(f"  features {split}: {i + 1}/{len(d['imgs'])}", flush=True)
    if names is None:
        raise RuntimeError(f"No usable rows in {split}")
    return SplitData(
        x=np.asarray(rows, dtype=np.float32),
        y_width=np.asarray(y, dtype=np.float32),
        raw_width=np.asarray(raw, dtype=np.float32),
        src=np.asarray(src),
        feature_names=names,
    )


def rel_err(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1e-6)))


def bias(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) / np.maximum(gt, 1e-6)))


def eval_pred(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    return {
        "relerr": round(rel_err(pred, gt), 4),
        "bias": round(bias(pred, gt), 4),
        "mae_px": round(float(np.mean(np.abs(pred - gt))), 3),
    }


def quantile_ratio_fit(raw: np.ndarray, gt: np.ndarray) -> dict:
    m = raw > 0
    qs = np.quantile(raw[m], [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(raw[m], qs)
    ratios = []
    for k in range(5):
        sel = bins == k
        ratios.append(float(np.median(gt[m][sel] / raw[m][sel])) if sel.sum() > 5 else 1.0)
    return {"qs": qs, "ratios": np.asarray(ratios, dtype=float)}


def quantile_ratio_predict(model: dict, raw: np.ndarray) -> np.ndarray:
    bins = np.digitize(raw, model["qs"])
    return np.where(raw > 0, raw * model["ratios"][bins], raw)


def make_model_specs():
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge_logwidth": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "hgb_logwidth": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.02,
            random_state=0,
        ),
        "extra_trees_logwidth": ExtraTreesRegressor(
            n_estimators=48,
            max_depth=10,
            min_samples_leaf=4,
            random_state=0,
            n_jobs=4,
        ),
    }


def fit_models(train: SplitData, val: SplitData):
    y_tr = np.log1p(train.y_width)
    specs = make_model_specs()
    fitted = {}
    val_scores = {}
    for name, model in specs.items():
        model.fit(train.x, y_tr)
        pred = np.expm1(model.predict(val.x))
        fitted[name] = model
        val_scores[name] = eval_pred(pred, val.y_width)
    return fitted, val_scores


def fit_one_model(name: str, data: SplitData):
    specs = make_model_specs()
    if name not in specs:
        raise KeyError(name)
    model = specs[name]
    model.fit(data.x, np.log1p(data.y_width))
    return model


def per_source(pred: np.ndarray, gt: np.ndarray, src: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for s in sorted(set(src)):
        m = src == s
        out[str(s)] = eval_pred(pred[m], gt[m])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--save", type=Path, default=CKPT)
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    print("=== M2 v2-a GaugeHead-Tiny tabular specialist ===", flush=True)
    if FEATURE_CACHE.exists() and not args.rebuild_features:
        print(f"loading feature cache: {FEATURE_CACHE}", flush=True)
        z = np.load(FEATURE_CACHE, allow_pickle=True)
        names = [str(x) for x in z["feature_names"]]
        train = SplitData(z["train_x"], z["train_y"], z["train_raw"], z["train_src"], names)
        val = SplitData(z["val_x"], z["val_y"], z["val_raw"], z["val_src"], names)
        test = SplitData(z["test_x"], z["test_y"], z["test_raw"], z["test_src"], names)
    else:
        print(f"building feature cache under {CACHE}", flush=True)
        train = load_split("train")
        val = load_split("val")
        test = load_split("test")
        np.savez_compressed(
            FEATURE_CACHE,
            feature_names=np.asarray(train.feature_names),
            train_x=train.x,
            train_y=train.y_width,
            train_raw=train.raw_width,
            train_src=train.src,
            val_x=val.x,
            val_y=val.y_width,
            val_raw=val.raw_width,
            val_src=val.src,
            test_x=test.x,
            test_y=test.y_width,
            test_raw=test.raw_width,
            test_src=test.src,
        )
        print(f"saved feature cache: {FEATURE_CACHE}", flush=True)
    trainval = SplitData(
        x=np.concatenate([train.x, val.x], axis=0),
        y_width=np.concatenate([train.y_width, val.y_width], axis=0),
        raw_width=np.concatenate([train.raw_width, val.raw_width], axis=0),
        src=np.concatenate([train.src, val.src], axis=0),
        feature_names=train.feature_names,
    )

    print(f"cache: {CACHE}", flush=True)
    print(f"rows: train {len(train.y_width)} / val {len(val.y_width)} / test {len(test.y_width)}", flush=True)
    print(f"features: {len(train.feature_names)}", flush=True)

    qr_train = quantile_ratio_fit(train.raw_width, train.y_width)
    qr_val = quantile_ratio_predict(qr_train, val.raw_width)
    qr = quantile_ratio_fit(trainval.raw_width, trainval.y_width)
    qr_test = quantile_ratio_predict(qr, test.raw_width)

    fitted, val_scores = fit_models(train, val)
    val_scores["quantile_ratio_5bin"] = eval_pred(qr_val, val.y_width)
    selected = min(val_scores, key=lambda k: val_scores[k]["relerr"])
    print("validation:", flush=True)
    for name, score in val_scores.items():
        print(f"  {name:20s} relerr={score['relerr']:.3f} bias={score['bias']:+.3f}", flush=True)
    print(f"selected by val: {selected}", flush=True)

    test_scores = {
        "raw_sam3_mask_width": eval_pred(test.raw_width, test.y_width),
        "quantile_ratio_5bin": eval_pred(qr_test, test.y_width),
        "m2_neural_refiner_v1": {
            "relerr": 0.564,
            "bias": 0.5033,
            "note": "from experiments/results/m2_refiner.json; not recomputed here",
        },
    }
    test_predictions = {
        "raw_sam3_mask_width": test.raw_width,
        "quantile_ratio_5bin": qr_test,
    }
    for name, model in fitted.items():
        pred = np.expm1(model.predict(test.x))
        test_predictions[name] = pred
        test_scores[name] = eval_pred(pred, test.y_width)
        refit = fit_one_model(name, trainval)
        pred_refit = np.expm1(refit.predict(test.x))
        test_predictions[f"{name}_refit_trainval"] = pred_refit
        test_scores[f"{name}_refit_trainval"] = eval_pred(pred_refit, test.y_width)

    selected_refit_model = None
    if selected != "quantile_ratio_5bin":
        selected_refit_model = fit_one_model(selected, trainval)
        selected_pred = np.expm1(selected_refit_model.predict(test.x))
        selected_for_ckpt = selected_refit_model
        selected_name_for_ckpt = f"{selected}_refit_trainval"
    else:
        selected_pred = qr_test
        selected_for_ckpt = None
        selected_name_for_ckpt = selected

    result = {
        "protocol": {
            "task": "predict crack mask-derived width from SAM3 mask + grayscale image statistics",
            "data": "datasets/m2_cache train/val/test; test sources are held out from m2_refiner.py",
            "selection": "model family selected on train-source validation only; test labels not used",
            "feature_count": len(train.feature_names),
            "feature_names": train.feature_names,
            "old_success_bar": {
                "raw_relerr": 0.7302,
                "m2_neural_v1_relerr": 0.564,
                "quantile_5bin_relerr": 0.4804,
            },
        },
        "n": {
            "train": int(len(train.y_width)),
            "val": int(len(val.y_width)),
            "test": int(len(test.y_width)),
        },
        "validation": val_scores,
        "selected_by_val": selected,
        "selected_checkpoint_model": selected_name_for_ckpt,
        "test": test_scores,
        "selected_test_per_source": per_source(selected_pred, test.y_width, test.src),
        "interpretation": (
            "This is a tiny owned measurement head. It is publishable only if it beats the "
            "5-bin calibration on held-out sources and reduces per-source worst-case bias."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    with args.save.open("wb") as f:
        pickle.dump(
            {
                "selected": selected_name_for_ckpt,
                "model": selected_for_ckpt,
                "feature_names": train.feature_names,
                "quantile_ratio": qr,
                "result_json": str(args.out),
            },
            f,
        )
    print("test:", flush=True)
    for name, score in test_scores.items():
        if "relerr" in score:
            print(f"  {name:22s} relerr={score['relerr']:.3f} bias={score.get('bias', 0):+.3f}", flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"saved {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
