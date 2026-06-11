"""M2 v2-a — 로짓 iso-level을 폭 보정 노브로: mask_threshold 스윕.

근거 사슬: N3("잔여 bias의 원인은 마스크 두께") + D("SAM3 소프트 로짓 노출").
가설: thin 크랙에서 SAM3 마스크가 GT보다 두꺼움(+bias) → post_process의
mask_threshold(로짓 iso-level)를 높이면 마스크가 얇아짐 → train 소스에서
폭-최적 θ*를 선택하면 신경 refiner 없이 bias 교정.

프로토콜 (감사 규율): θ* 선택은 train(300 서브샘플)+val(149)에서만,
보고는 홀드아웃 test 소스(cfd/cracktree200/deepcrack). 1 forward → 9 threshold
post_process (재추론 없음). 합격선: 분위 보정 0.480 (m2_calibration_baseline).

실행: python experiments/m2v2_logit_threshold.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.m2_refiner import build_splits, load_gray_gt, width_of  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
SCORE_THR = 0.4
OUT = Path("experiments/results/m2v2_logit_threshold.json")


def widths_multi_threshold(rgb: np.ndarray):
    """1회 forward → 각 mask_threshold에서 union 마스크 폭."""
    import torch
    from gaugeanything import segmenters as S
    if "model" not in S._SAM3:
        S.segment_sam3(rgb, "crack")  # 모델 로드 트리거
    proc, model, dev = S._SAM3["proc"], S._SAM3["model"], S._SAM3["dev"]
    from PIL import Image as PILImage
    pil = PILImage.fromarray(rgb)
    inputs = proc(images=pil, text="crack", return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    ws = {}
    for th in THRESHOLDS:
        res = proc.post_process_instance_segmentation(
            out, threshold=SCORE_THR, mask_threshold=th, target_sizes=[pil.size[::-1]])[0]
        masks = [np.asarray(m.cpu()).squeeze().astype(bool) for m in res["masks"]]
        union = np.any(masks, 0) if masks else np.zeros(rgb.shape[:2], bool)
        ws[th] = width_of(union)
    return ws


def metrics(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    rel = np.abs(pred - gt) / np.maximum(gt, 1e-6)
    b = (pred - gt) / np.maximum(gt, 1e-6)
    return float(rel.mean()), float(b.mean())


def main():
    rng = np.random.default_rng(1)
    train_pool, val_pool, test_pool = build_splits(seed=0)
    sel_items = [train_pool[i] for i in rng.permutation(len(train_pool))[:300]] + val_pool
    print(f"=== M2 v2-a 로짓 threshold 스윕 ===  선택셋 {len(sel_items)} / test {len(test_pool)}")

    def collect(items, name):
        rows = []
        t0 = time.time()
        for k, (ip, mp, src) in enumerate(items):
            g, gt = load_gray_gt(ip, mp)
            if gt.sum() < 30:
                continue
            w_gt = width_of(gt)
            if w_gt <= 0:
                continue
            rgb = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
            ws = widths_multi_threshold(rgb)
            rows.append({"src": src, "w_gt": w_gt, **{f"w@{t}": ws[t] for t in THRESHOLDS}})
            if (k + 1) % 100 == 0:
                print(f"  {name}: {k+1}/{len(items)} ({time.time()-t0:.0f}s)")
        return rows

    sel = collect(sel_items, "선택셋")
    te = collect(test_pool, "test")

    # θ* 선택 (선택셋 rel_err 최소)
    sel_scores = {}
    for th in THRESHOLDS:
        pred = [r[f"w@{th}"] for r in sel]
        gt = [r["w_gt"] for r in sel]
        sel_scores[th] = metrics(pred, gt)
    th_star = min(sel_scores, key=lambda t: sel_scores[t][0])
    print("\n선택셋 스윕:", {t: f"rel {v[0]:.3f}/bias {v[1]:+.3f}" for t, v in sel_scores.items()})
    print(f"θ* = {th_star}")

    # 분위 보정 (θ* 폭, 선택셋 적합)
    wr = np.array([r[f"w@{th_star}"] for r in sel]); wg = np.array([r["w_gt"] for r in sel])
    m = wr > 0
    qs = np.quantile(wr[m], [0.2, 0.4, 0.6, 0.8])
    ratios = []
    bins = np.digitize(wr[m], qs)
    for k in range(5):
        s = bins == k
        ratios.append(float(np.median(wg[m][s] / wr[m][s])) if s.sum() > 5 else 1.0)

    def q_cal(w):
        w = np.asarray(w, float)
        return np.where(w > 0, w * np.array(ratios)[np.digitize(w, qs)], w)

    # test 보고
    res = {"theta_star": th_star, "sweep_selection": {str(t): {"rel": round(v[0], 4), "bias": round(v[1], 4)}
                                                     for t, v in sel_scores.items()},
           "quantile_ratios": [round(r, 3) for r in ratios], "n_test": len(te)}
    gt_te = [r["w_gt"] for r in te]
    table = {
        "default@0.5": [r["w@0.5"] for r in te],
        f"theta*@{th_star}": [r[f"w@{th_star}"] for r in te],
        f"theta*+qcal": q_cal([r[f"w@{th_star}"] for r in te]),
        "default+qcal(0.5비교용)": None,
    }
    print(f"\n{'방법':<22}{'test rel_err':>13}{'bias':>9}")
    print("-" * 46)
    for name, pred in table.items():
        if pred is None:
            continue
        r, b = metrics(pred, gt_te)
        res[name] = {"rel": round(r, 4), "bias": round(b, 4)}
        print(f"{name:<22}{r:>13.3f}{b:>+9.3f}")
    # per-source @ best
    best_name = min((k for k in res if isinstance(res[k], dict) and "rel" in res[k]),
                    key=lambda k: res[k]["rel"])
    per_src = {}
    for s in sorted({r["src"] for r in te}):
        idx = [i for i, r in enumerate(te) if r["src"] == s]
        pred = np.asarray(table[f"theta*+qcal"])[idx]
        gtv = np.asarray(gt_te)[idx]
        rr, bb = metrics(pred, gtv)
        per_src[s] = {"rel": round(rr, 4), "bias": round(bb, 4), "n": len(idx)}
    res["per_source_theta*+qcal"] = per_src
    print("per-source (θ*+qcal):", per_src)
    print("\n앵커: 분위보정@0.5 = 0.480 (m2_calibration_baseline) · M2 v1 신경 = 0.564")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"결과 저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
