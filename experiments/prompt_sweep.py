"""SAM3 프롬프트 민감도 스윕 (RIGOR_AUDIT A5).

zero-shot 수치가 프롬프트 선택의 함수인지 정량화. 도메인별 4개 프롬프트 × crack-only mIoU.
실행: python experiments/prompt_sweep.py --n 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_bench import find_pairs, iou_f1, load_gray_mask  # noqa: E402
from experiments.gauge_multidomain import load_pair, mt_pairs  # noqa: E402
from gaugeanything.segmenters import segment_sam3  # noqa: E402

SWEEP = {
    "concrete_crack": (["crack", "cracks", "fracture", "thin dark crack"], "crackseg9k"),
    "mt_crack": (["crack", "cracks", "scratch", "dark line"], "mt:Crack"),
    "mt_blowhole": (["hole", "pit", "blowhole", "small dark spot"], "mt:Blowhole"),
}


def get_samples(src, n, rng):
    if src == "crackseg9k":
        pairs = [p for p in find_pairs(Path("datasets/crackseg9k"))]
        idx = rng.permutation(len(pairs))[: n * 3]
        out = []
        for i in idx:
            img, gt = load_gray_mask(pairs[i][0], pairs[i][1])
            if gt.sum() >= 30:
                out.append((img, gt))
            if len(out) >= n:
                break
        return out
    defect = src.split(":")[1]
    pairs = mt_pairs(defect)
    idx = rng.permutation(len(pairs))[:n]
    return [(im, gt) for im, gt in (load_pair(*pairs[i][:2]) for i in idx) if gt.sum() >= 30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print("=== SAM3 프롬프트 민감도 스윕 ===")
    out = {}
    for dom, (prompts, src) in SWEEP.items():
        samples = get_samples(src, args.n, rng)
        row = {}
        for pr in prompts:
            ious = []
            for img, gt in samples:
                insts = segment_sam3(img, pr, threshold=0.4)
                mask = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(gt.shape, bool)
                ious.append(iou_f1(mask, gt)[0])
            row[pr] = round(float(np.mean(ious)), 4)
        vals = list(row.values())
        out[dom] = {"per_prompt": row, "mean": round(float(np.mean(vals)), 4),
                    "best": round(max(vals), 4), "spread": round(max(vals) - min(vals), 4),
                    "n": len(samples)}
        print(f"\n[{dom}] n={len(samples)}")
        for pr, v in row.items():
            print(f"  {pr!r:22s} mIoU={v:.3f}")
        print(f"  → mean {out[dom]['mean']:.3f} · best {out[dom]['best']:.3f} · spread {out[dom]['spread']:.3f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/prompt_sweep.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n결과 저장: experiments/results/prompt_sweep.json")
    print("주: spread가 크면 zero-shot 수치는 프롬프트 의존 — 보고 시 mean±spread 병기 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
