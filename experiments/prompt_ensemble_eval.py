"""프롬프트 앙상블 검증 — 동의어 붕괴를 구조하는가.

시나리오: 사용자가 붕괴 동의어("fracture", "pit")를 입력 → 단일 프롬프트는 0.0,
PROMPT_SETS 경유 앙상블은 검증 프롬프트로 회복하는지 측정. best 단일과의 차이도 보고.

실행: python experiments/prompt_ensemble_eval.py --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.prompt_sweep import get_samples  # noqa: E402
from experiments.gauge_bench import iou_f1  # noqa: E402
from gaugeanything.segmenters import segment_sam3, segment_sam3_ensemble  # noqa: E402

# (도메인, 샘플 소스, 붕괴 동의어, best 단일 프롬프트)
CASES = [
    ("concrete_crack", "crackseg9k", "fracture", "cracks"),
    ("mt_blowhole", "mt:Blowhole", "pit", "hole"),
]


def miou(samples, fn):
    ious = []
    for img, gt in samples:
        insts = fn(img)
        mask = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(gt.shape, bool)
        ious.append(iou_f1(mask, gt)[0])
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print("=== 프롬프트 앙상블 — 동의어 붕괴 구조 검증 ===")
    out = {}
    for dom, src, broken, best in CASES:
        samples = get_samples(src, args.n, rng)
        m_broken = miou(samples, lambda im: segment_sam3(im, broken, threshold=0.4))
        m_best = miou(samples, lambda im: segment_sam3(im, best, threshold=0.4))
        m_ens = miou(samples, lambda im: segment_sam3_ensemble(im, broken, threshold=0.4))
        out[dom] = {"n": len(samples), "broken_prompt": broken,
                    "single_broken": round(m_broken, 4), "single_best": round(m_best, 4),
                    "ensemble_via_broken": round(m_ens, 4)}
        print(f"\n[{dom}] n={len(samples)}  사용자 입력='{broken}'")
        print(f"  단일({broken!r}):      {m_broken:.3f}   ← 붕괴")
        print(f"  단일 best({best!r}):  {m_best:.3f}")
        print(f"  앙상블('{broken}'→셋): {m_ens:.3f}   ← 구조 여부")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/prompt_ensemble.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n결과 저장: experiments/results/prompt_ensemble.json")
    print("판정: ensemble ≈ best 면 어휘 취약성이 매핑 레이어로 해소됨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
