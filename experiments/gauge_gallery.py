"""정성 결과 갤러리 — 여러 크랙 소스에 GaugeAnything 적용 (프로젝트 페이지용).

SAM3를 1회만 로드하고 소스별 1장씩 3패널(입력|분할|폭 히트맵) 생성.
정직하게 어려운 케이스(cracktree200, gaps384)도 포함.

실행: python experiments/gauge_gallery.py --out docs/assets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_demo import pick_image, render, width_heatmap  # noqa: E402
from gaugeanything.segmenters import segment_sam3  # noqa: E402

# (소스, 표시명, 가정 mm/px) — 다양한 표면 + 정직한 난이도 포함
SOURCES = [
    ("rissbilder", "콘크리트 벽 (Rissbilder)", 0.25),
    ("crack500", "포장 도로 (Crack500)", 0.50),
    ("deepcrack", "표면 크랙 (DeepCrack)", 0.20),
    ("cracktree200", "가는 크랙 (CrackTree) · 난이도↑", 0.15),
    ("cfd", "도로 표면 (CFD)", 0.30),
    ("volker", "구조물 (Volker)", 0.25),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/crackseg9k")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--prompt", default="crack")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    rows = []
    for src, label, mmpp in SOURCES:
        try:
            img_path = pick_image(Path(args.root), src)
        except (ValueError, FileNotFoundError):
            print(f"  ! {src} 이미지 없음 — 건너뜀"); continue
        img = np.array(Image.open(img_path).convert("RGB"))
        insts = segment_sam3(img, args.prompt, threshold=0.4)
        mask = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(img.shape[:2], bool)
        heat, stats = width_heatmap(mask) if mask.sum() > 20 else (np.zeros(img.shape[:2]), {"mean": 0, "max": 0, "p95": 0, "len": 0})
        op = out / f"gallery_{src}.png"
        render(img, mask, heat, stats, mmpp, op, args.prompt)
        rows.append((label, src, len(insts), stats["mean"] * mmpp, mmpp))
        print(f"  ✓ {src}: 인스턴스 {len(insts)}, 평균폭 {stats['mean']*mmpp:.2f}mm")

    print("\n=== 갤러리 요약 ===")
    for label, src, n, w, mmpp in rows:
        print(f"  {label:36s} 인스턴스 {n:2d}  평균폭 {w:.2f}mm (@{mmpp}mm/px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
