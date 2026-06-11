"""N3 — M2 v2 보정-only 베이스라인: 단순 보정이 신경 refiner를 따라잡는가.

감사 질문(RESEARCH_AUDIT N3): 폭 bias가 도메인 의존이라면, 신경 refiner(0.564) 대신
**train-source에서 적합한 단순 보정**으로 얼마나 회복되는가? 단순한 쪽이 비슷하면
신경 refiner는 과대주장 — 정직성 체크.

보정 후보 (전부 train+val 소스에서만 적합, test 소스는 미노출):
  A. 전역 아핀: w_cal = a·w_raw + b
  B. 폭-구간(quantile bin) 보정: raw 폭 분위별 배율
  C. 이미지 통계 조건부 아핀: [w_raw, mask 면적/스켈레톤 길이 통계] 선형회귀

데이터: m2_refiner의 캐시(npz) 재사용 — raw SAM3 마스크에서 폭 재계산.
실행: python experiments/m2_calibration_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.m2_refiner import CACHE, width_of  # noqa: E402

OUT = Path("experiments/results/m2_calibration_baseline.json")


def widths_from_cache(split):
    d = np.load(CACHE / f"{split}.npz", allow_pickle=True)
    rows = []
    for i in range(len(d["imgs"])):
        sam = d["sams"][i].astype(bool)
        gt = d["gts"][i].astype(bool)
        w_gt = width_of(gt)
        if w_gt <= 0:
            continue
        w_raw = width_of(sam)
        # 조건부 특징: 마스크 면적, 스켈레톤 길이 근사(면적/폭), 마스크 유무
        area = float(sam.sum())
        feat_len = area / max(w_raw, 1e-6)
        rows.append({"w_raw": w_raw, "w_gt": w_gt, "area": area,
                     "len": feat_len, "src": str(d["srcs"][i])})
    return rows


def rel_err(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1e-6)))


def bias(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    return float(np.mean((pred - gt) / np.maximum(gt, 1e-6)))


def main():
    tr = widths_from_cache("train") + widths_from_cache("val")
    te = widths_from_cache("test")
    print(f"=== N3 보정-only 베이스라인 ===  train+val n={len(tr)} / test n={len(te)}")
    wr_tr = np.array([r["w_raw"] for r in tr]); wg_tr = np.array([r["w_gt"] for r in tr])
    wr_te = np.array([r["w_raw"] for r in te]); wg_te = np.array([r["w_gt"] for r in te])
    # 검출 실패(w_raw=0)는 보정 불가 — raw와 동일 취급, 비율 보고
    miss = float(np.mean(wr_te <= 0))

    res = {"n_train": len(tr), "n_test": len(te), "test_miss_rate": round(miss, 3),
           "anchors": {"raw_relerr": round(rel_err(wr_te, wg_te), 4),
                       "raw_bias": round(bias(wr_te, wg_te), 4),
                       "m2_neural_relerr": 0.564, "m2_neural_note": "m2_refiner.json 공식"}}
    print(f"raw SAM3: rel_err {res['anchors']['raw_relerr']:.3f}  bias {res['anchors']['raw_bias']:+.3f}  "
          f"(미검출 {miss*100:.0f}%)")

    # A. 전역 아핀 (train의 w_raw>0만으로 적합)
    m = wr_tr > 0
    A = np.stack([wr_tr[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, wg_tr[m], rcond=None)[0]
    predA = np.where(wr_te > 0, np.maximum(0, a * wr_te + b), wr_te)
    res["A_global_affine"] = {"a": round(float(a), 4), "b": round(float(b), 3),
                              "relerr": round(rel_err(predA, wg_te), 4),
                              "bias": round(bias(predA, wg_te), 4)}

    # B. 분위 bin 배율 (train에서 raw 폭 5분위별 median(gt/raw))
    qs = np.quantile(wr_tr[m], [0.2, 0.4, 0.6, 0.8])
    ratios = []
    binsA = np.digitize(wr_tr[m], qs)
    for k in range(5):
        sel = binsA == k
        ratios.append(float(np.median(wg_tr[m][sel] / wr_tr[m][sel])) if sel.sum() > 5 else 1.0)
    binsT = np.digitize(wr_te, qs)
    predB = np.where(wr_te > 0, wr_te * np.array(ratios)[binsT], wr_te)
    res["B_quantile_ratio"] = {"ratios": [round(r, 3) for r in ratios],
                               "relerr": round(rel_err(predB, wg_te), 4),
                               "bias": round(bias(predB, wg_te), 4)}

    # C. 특징 조건부 아핀 [w_raw, log-area, log-len]
    def feats(rows, msk=None):
        X = np.stack([[r["w_raw"], np.log1p(r["area"]), np.log1p(r["len"])] for r in rows])
        return X
    Xtr = feats(tr)[m]; Xte = feats(te)
    A2 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    coef = np.linalg.lstsq(A2, wg_tr[m], rcond=None)[0]
    predC_all = np.maximum(0, np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ coef)
    predC = np.where(wr_te > 0, predC_all, wr_te)
    res["C_feature_affine"] = {"relerr": round(rel_err(predC, wg_te), 4),
                               "bias": round(bias(predC, wg_te), 4)}

    print(f"\n{'방법':<22}{'test rel_err':>14}{'bias':>10}")
    print("-" * 48)
    print(f"{'raw SAM3':<22}{res['anchors']['raw_relerr']:>14.3f}{res['anchors']['raw_bias']:>+10.3f}")
    for k in ("A_global_affine", "B_quantile_ratio", "C_feature_affine"):
        print(f"{k:<22}{res[k]['relerr']:>14.3f}{res[k]['bias']:>+10.3f}")
    print(f"{'M2 신경 refiner(공식)':<21}{0.564:>14.3f}{'+0.503':>10}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {OUT}")
    print("판정 기준: 단순 보정이 0.564에 근접하면 신경 refiner 기여는 제한적 — 정직 보고")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
