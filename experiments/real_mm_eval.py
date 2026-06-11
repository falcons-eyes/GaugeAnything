"""실측 mm 검증 — 캘리퍼 GT vs 파이프라인 예측 (docs/CAPTURE_PROTOCOL.md).

captures/manifest.csv + 이미지 → inspect_soft(PlaneScale) → 폭(mm) 예측 →
캘리퍼 대비 MAE(mm)·상대오차 분포·±10%/±20% 합격률.

실행: python experiments/real_mm_eval.py --captures captures/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaugeanything.router import inspect_soft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="captures")
    ap.add_argument("--segmenter", default="sam3_ensemble")
    args = ap.parse_args()
    cap = Path(args.captures)
    mf = cap / "manifest.csv"
    if not mf.exists():
        print(f"✗ {mf} 없음 — docs/CAPTURE_PROTOCOL.md의 manifest 형식 참조")
        return 1

    from PIL import Image
    rows, errs_rel, errs_abs = [], [], []
    with open(mf) as f:
        for row in csv.DictReader(f):
            ip = cap / row["image"]
            if not ip.exists():
                print(f"  ! 이미지 없음: {ip}"); continue
            gt = float(row["caliper_mm_max"])
            img = np.array(Image.open(ip).convert("RGB"))
            atoms = inspect_soft(img, row.get("prompt") or row["defect_type"],
                                 segmenter=args.segmenter,
                                 marker_size_mm=float(row["marker_size_mm"]))
            # 가장 큰 thin 인스턴스의 p95 폭(최대폭 근사) — 캘리퍼 '최대 폭' 측정 규약과 정렬
            thin = [a for a in atoms if a.metrics.get("kind") == "thin"]
            if not thin:
                print(f"  ! {row['image']}: thin 인스턴스 미검출 (regime: "
                      f"{[a.regime for a in atoms]})"); continue
            a = max(thin, key=lambda x: x.metrics.get("area", 0))
            pred = a.metrics.get("p95", a.metrics.get("width_mean", 0.0))
            e_abs, e_rel = abs(pred - gt), abs(pred - gt) / max(gt, 1e-6)
            errs_abs.append(e_abs); errs_rel.append(e_rel)
            rows.append({"image": row["image"], "gt_mm": gt, "pred_mm": round(pred, 3),
                         "err_mm": round(e_abs, 3), "err_rel": round(e_rel, 3)})
            print(f"  {row['image']:24s} GT {gt:6.2f}mm → 예측 {pred:6.2f}mm "
                  f"(오차 {e_rel*100:5.1f}%)")

    if not rows:
        print("평가 가능 샘플 없음"); return 1
    rel = np.array(errs_rel)
    summary = {"n": len(rows), "MAE_mm": round(float(np.mean(errs_abs)), 3),
               "rel_err_mean": round(float(rel.mean()), 3),
               "rel_err_median": round(float(np.median(rel)), 3),
               "pass@10pct": round(float((rel <= 0.10).mean()), 3),
               "pass@20pct": round(float((rel <= 0.20).mean()), 3)}
    print(f"\n{'='*56}\n실측 검증 (n={summary['n']}): MAE {summary['MAE_mm']}mm · "
          f"상대오차 중앙값 {summary['rel_err_median']*100:.1f}% · "
          f"±10% 합격 {summary['pass@10pct']*100:.0f}% / ±20% {summary['pass@20pct']*100:.0f}%")
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "real_mm_eval.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/real_mm_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
