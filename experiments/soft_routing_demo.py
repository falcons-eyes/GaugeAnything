"""Regime 라우팅 검증 — 각 결함 타입이 올바른 regime으로 분기되는가.

기대: crack/blowhole → sharp/fuzzy(마스크 있음), fray → fuzzy/field, uneven → field.
실행: python experiments/soft_routing_demo.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_multidomain import load_pair, mt_pairs  # noqa: E402
from gaugeanything.router import inspect_soft  # noqa: E402

CASES = [("Crack", "crack"), ("Blowhole", "hole"), ("Break", "crack"),
         ("Fray", "scratch"), ("Uneven", "stain")]


def main():
    print("=== Regime 라우팅 검증 (inspect_soft) ===")
    rng = np.random.default_rng(0)
    for defect, prompt in CASES:
        pairs = mt_pairs(defect)
        if not pairs:
            print(f"  ! MT_{defect} 없음"); continue
        regimes, lines = [], []
        for k in rng.permutation(len(pairs))[:8]:
            img, _ = load_pair(*pairs[k][:2])
            atoms = inspect_soft(img, prompt, segmenter="sam3", mm_per_px=0.2)
            for a in atoms:
                regimes.append(a.regime)
            if atoms and len(lines) < 1:
                lines.append(atoms[0].summary())
        dist = dict(Counter(regimes))
        print(f"\n[MT_{defect}] prompt='{prompt}'  regime 분포: {dist}")
        for ln in lines:
            print(f"   예: {ln}")
    print("\n주: uneven→field, fray→fuzzy/field, crack/blowhole→sharp/fuzzy 면 라우팅 정상")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
