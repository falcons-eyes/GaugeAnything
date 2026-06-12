"""E-loop 최종 — 멀티-인스턴스 평가의 정식 스크립트 (ad-hoc 인라인 → 재현 가능 승격).

계측기의 올바른 출력 = 장면의 모든 크랙 인스턴스 각각 측정 (Inspection Atoms).
평가: 주석 크랙이 출력 인스턴스 중 하나로 커버되는가(recall) + 매칭 인스턴스의 폭 오차.

실행: python experiments/krkcmd_multiinstance_eval.py --stages 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.krkcmd_image_eval import N_P, load_rows, load_stack, to_gray, refine_y  # noqa: E402
from experiments.krkcmd_loop_pathselect import multi_component_paths, SNAP  # noqa: E402
from experiments.krkcmd_signal_width import train_profile_cnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CMd_0.23_2mths")
    ap.add_argument("--image", default="1")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--col-tol", type=int, default=15)
    args = ap.parse_args()

    print("=== 멀티-인스턴스 평가 (정식) ===")
    predict, table_mae = train_profile_cnn(epochs=12)
    pages = load_stack(args.series, args.image)
    rows_all = load_rows(args.series, args.image)

    near = tot = 0
    errs = []
    per_stage = []
    for stage in sorted({r["stage"] for r in rows_all})[: args.stages]:
        rows = [r for r in rows_all if r["stage"] == stage]
        gray = to_gray(pages[stage - 1])
        H, W = gray.shape
        paths = multi_component_paths(gray, min_span=0.05)
        s_near = s_tot = 0
        for r in rows:
            c = r["gx"]
            if c >= W:
                continue
            y0, sc = refine_y(gray[:, c], r["x"], r["gy"] - 250, win=60)
            if sc < 0.95:
                continue
            yo = y0 + 250
            tot += 1; s_tot += 1
            best = None
            for path in paths:
                cols = np.array(sorted(path))
                if not len(cols):
                    continue
                j = int(np.argmin(np.abs(cols - c)))
                if abs(int(cols[j]) - c) > args.col_tol:
                    continue
                yh = path[int(cols[j])]
                lo = max(0, yh - SNAP); hi = min(H, yh + SNAP + 1)
                yc = lo + int(np.argmin(gray[lo:hi, c]))
                if abs(yc - yo) <= 80 and (best is None or abs(yc - yo) < abs(best - yo)):
                    best = yc
            if best is not None:
                near += 1; s_near += 1
                a = max(0, best - N_P // 2)
                p = gray[a: a + N_P, c]
                if len(p) == N_P:
                    errs.append(abs(float(predict([p])[0]) - r["man"]))
        per_stage.append({"stage": stage, "recall": round(s_near / max(s_tot, 1), 3),
                          "n_paths": len(paths)})
        print(f"[stage {stage}] paths={len(paths)} recall {s_near}/{s_tot}")

    rec = near / max(tot, 1)
    mae = float(np.mean(errs)); med = float(np.median(errs))
    print(f"\n멀티-인스턴스: recall {near}/{tot} = {rec*100:.0f}% · "
          f"폭 MAE {mae:.1f}μm 중앙 {med:.1f}μm (n={len(errs)})")
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "krkcmd_multiinstance_eval.json").write_text(json.dumps(
        {"recall": round(rec, 3), "MAE_um": round(mae, 1), "median_um": round(med, 1),
         "n": len(errs), "table_cnn_MAE": round(table_mae, 1), "per_stage": per_stage},
        indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/krkcmd_multiinstance_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
