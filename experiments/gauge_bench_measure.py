"""Gauge-Bench 측정 평가 — 분할을 넘어 '측정 정확도'를 1급 지표로.

GaugeAnything 고유 차별점(어떤 FM도 안 하는 측정)을 실데이터로 평가한다.
mm GT는 공개 데이터에 없으므로(VISION_DESIGN §6.2), GT 마스크에서 잰 폭/길이를
'참값'으로 삼아 예측 마스크에서 잰 값과 비교한다:

  question: 세그멘터가 좋은 마스크를 줄 때, 거기서 잰 폭이 GT 마스크 폭을 얼마나 추종하는가?
  metric  : width MAE(px), width 상대오차, length 상대오차 — 측정 충실도

이는 "분할 IoU는 같아도 측정 오차는 다를 수 있다"를 드러낸다 (mIoU만으론 부족).

실행: python experiments/gauge_bench_measure.py --n 120 --segmenters adaptive sam3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_bench import build_sam3, find_pairs, iou_f1, load_gray_mask  # noqa: E402
from gaugeanything.baselines import BASELINES  # noqa: E402
from gaugeanything.geometry import measure_thin  # noqa: E402


def width_length(mask: np.ndarray):
    """마스크 → (mean_width_px, p95_width_px, length_px). 빈 마스크는 모두 0."""
    if mask.sum() < 20:
        return 0.0, 0.0, 0.0
    g = measure_thin(mask)
    return g.width_mean, g.width_p95, g.length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/crackseg9k")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--segmenters", nargs="+", default=["adaptive", "sam3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gt", type=int, default=200, help="측정 대상 최소 GT 픽셀")
    args = ap.parse_args()

    print("=== Gauge-Bench 측정 평가 (width/length 충실도) ===")
    pairs = find_pairs(Path(args.root))
    by_src = defaultdict(list)
    for p in pairs:
        by_src[p[2]].append(p)
    rng = np.random.default_rng(args.seed)
    per_src = max(1, args.n // len(by_src))
    sample = []
    for s, v in by_src.items():
        idx = rng.permutation(len(v))[:per_src]
        sample += [v[i] for i in idx]
    rng.shuffle(sample)
    sample = sample[:args.n]
    print(f"평가 샘플: {len(sample)}\n")

    segfns = {}
    for name in args.segmenters:
        if name in BASELINES:
            segfns[name] = BASELINES[name]
        elif name == "sam3":
            segfns[name] = build_sam3()

    # seg -> 누적
    acc = {n: {"iou": [], "werr": [], "wrel": [], "lrel": [],
               "gt_w": [], "pred_w": []} for n in segfns}
    timing = defaultdict(float)

    for k, (img_p, mask_p, src) in enumerate(sample):
        img, gt = load_gray_mask(img_p, mask_p)
        if gt.sum() < args.min_gt:   # 측정은 크랙 있는 이미지에서만
            continue
        gt_w, gt_w95, gt_len = width_length(gt)
        if gt_w == 0:
            continue
        for name, fn in segfns.items():
            t0 = time.time()
            try:
                pred = fn(img)
            except Exception as e:
                if k == 0:
                    print(f"  ! {name}: {str(e)[:100]}")
                continue
            timing[name] += time.time() - t0
            iou, _ = iou_f1(pred, gt)
            pw, pw95, plen = width_length(pred)
            acc[name]["iou"].append(iou)
            acc[name]["gt_w"].append(gt_w)
            acc[name]["pred_w"].append(pw)
            acc[name]["werr"].append(abs(pw - gt_w))
            acc[name]["wrel"].append(abs(pw - gt_w) / max(gt_w, 1e-6))
            acc[name]["lrel"].append(abs(plen - gt_len) / max(gt_len, 1e-6))
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(sample)} ...")

    print(f"\n{'='*72}")
    print(f"{'segmenter':<12}{'mIoU':>7}{'width MAE(px)':>14}{'width 상대err':>14}{'len 상대err':>13}")
    print("-" * 72)
    summary = {}
    for name in segfns:
        a = acc[name]
        if not a["iou"]:
            continue
        s = {
            "n": len(a["iou"]),
            "mIoU": round(float(np.mean(a["iou"])), 4),
            "width_mae_px": round(float(np.mean(a["werr"])), 3),
            "width_rel_err": round(float(np.mean(a["wrel"])), 3),
            "length_rel_err": round(float(np.mean(a["lrel"])), 3),
            "gt_width_mean_px": round(float(np.mean(a["gt_w"])), 2),
            "pred_width_mean_px": round(float(np.mean(a["pred_w"])), 2),
            "sec_per_img": round(timing[name] / max(len(a["iou"]), 1), 3),
        }
        summary[name] = s
        print(f"{name:<12}{s['mIoU']:>7.3f}{s['width_mae_px']:>14.2f}"
              f"{s['width_rel_err']*100:>12.1f}%{s['length_rel_err']*100:>11.1f}%")
        print(f"             GT폭 평균 {s['gt_width_mean_px']:.1f}px → 예측폭 {s['pred_width_mean_px']:.1f}px")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "gauge_bench_measure.json", "w") as f:
        json.dump({"n": len(sample), "results": summary}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: experiments/results/gauge_bench_measure.json")
    print("주: mIoU가 높아도 width 상대오차가 크면 '측정용'으론 부족 — 측정 1급 지표의 의의")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
