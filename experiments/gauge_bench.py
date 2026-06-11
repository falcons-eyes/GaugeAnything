"""Gauge-Bench v0 — 크랙 분할 평가 하네스 (CrackSeg9k).

센서 트랙에서 배운 규율 적용 (GAP_ANALYSIS):
  - 베이스라인 사다리: 고전(otsu/adaptive/frangi/blackhat) vs SAM3
  - 소스 confound 회피: CrackSeg9k 파일 접두사 = 원본 소스 → cross-source 분할 가능
  - 상업 라이선스: NC 소스(DeepCrack/GAPs) 제외 옵션
  - 측정 1급 지표: 분할 IoU + 폭 측정 오차(향후 GT 있을 때)
  - 결과 JSON 저장

CrackSeg9k 구조: Final-Dataset-Vol{1,2}/Images*/  +  Final_Masks/Masks/
파일명 접두사: CRACK500_, GAPS384_, DeepCrack_, Rissbilder_, Volker_, CFD_, noncrack_ ...

실행: python experiments/gauge_bench.py --n 200 --segmenters otsu frangi sam3
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
from gaugeanything.baselines import BASELINES  # noqa: E402

# 상업 비클린 소스 (원본 NC) — 상업 트랙에서 제외
NC_PREFIXES = ("DeepCrack", "GAPS", "GAPs")


def source_of(fname: str) -> str:
    """파일명 접두사 → 원본 소스 데이터셋."""
    base = fname.split("_")[0]
    return base.lower()


def find_pairs(root: Path, commercial_only: bool = False):
    """(이미지경로, 마스크경로, 소스) 리스트. 이미지-마스크 파일명 매칭."""
    masks = {}
    for m in root.rglob("Final_Masks/Masks/*.png"):
        masks[m.stem] = m
    pairs = []
    for img in root.rglob("Images*/*"):
        if img.suffix.lower() not in (".jpg", ".png", ".jpeg"):
            continue
        if img.stem in masks:
            if commercial_only and img.stem.startswith(NC_PREFIXES):
                continue
            pairs.append((img, masks[img.stem], source_of(img.stem)))
    return pairs


def iou_f1(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    iou = inter / union if union else (1.0 if p.sum() == g.sum() == 0 else 0.0)
    prec = inter / p.sum() if p.sum() else 0.0
    rec = inter / g.sum() if g.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(iou), float(f1)


def load_gray_mask(img_path: Path, mask_path: Path):
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    gt = np.array(Image.open(mask_path).convert("L")) > 127
    return img, gt


def build_sam3(threshold=0.4):
    """SAM3 세그멘터 클로저. probe로 확인된 API 사용 (segmenters.segment_sam3)."""
    from gaugeanything.segmenters import segment_sam3

    def fn(image):
        insts = segment_sam3(image, prompt="crack", threshold=threshold)
        if not insts:
            return np.zeros(image.shape[:2], bool)
        return np.any([i.mask for i in insts], axis=0)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/crackseg9k")
    ap.add_argument("--n", type=int, default=150, help="시드당 평가 이미지 수 (소스 균형)")
    ap.add_argument("--segmenters", nargs="+", default=["otsu", "adaptive", "frangi", "blackhat"])
    ap.add_argument("--commercial-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="평가 시드 수 (mean±std)")
    args = ap.parse_args()

    print("=== Gauge-Bench v1 — 크랙 분할 (crack/noncrack 분리, 시드 평균) ===")
    pairs = find_pairs(Path(args.root), args.commercial_only)
    by_src = defaultdict(list)
    for p in pairs:
        by_src[p[2]].append(p)
    print(f"총 쌍: {len(pairs)} | 소스 수: {len(by_src)} | 시드: {args.seeds}")

    segfns = {}
    for name in args.segmenters:
        if name in BASELINES:
            segfns[name] = BASELINES[name]
        elif name == "sam3":
            segfns[name] = build_sam3()
        else:
            print(f"  ! 알 수 없는 세그멘터: {name}")

    # seed -> seg -> 누적
    crack_miou = defaultdict(list)   # seg -> [seed별 crack-only mIoU]
    fp_clean = defaultdict(list)     # seg -> [seed별 noncrack 무탐(클린) 비율]
    per_src_last = {}
    timing = defaultdict(float); count = defaultdict(int)

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        per_src_n = max(1, args.n // len(by_src))
        sample = []
        for s, v in by_src.items():
            idx = rng.permutation(len(v))[:per_src_n]
            sample += [v[i] for i in idx]
        rng.shuffle(sample); sample = sample[:args.n]

        ious = defaultdict(list); clean = defaultdict(list)
        psrc = defaultdict(lambda: defaultdict(list))
        for img_p, mask_p, src in sample:
            img, gt = load_gray_mask(img_p, mask_p)
            is_crack = gt.sum() >= 30
            for name, fn in segfns.items():
                t0 = time.time()
                try:
                    pred = fn(img)
                except Exception as e:
                    print(f"  ! {name}: {str(e)[:100]}"); continue
                timing[name] += time.time() - t0; count[name] += 1
                if is_crack:
                    iou, _ = iou_f1(pred, gt)
                    ious[name].append(iou); psrc[name][src].append(iou)
                else:
                    clean[name].append(1.0 if pred.sum() < 50 else 0.0)
        for name in segfns:
            if ious[name]:
                crack_miou[name].append(float(np.mean(ious[name])))
            if clean[name]:
                fp_clean[name].append(float(np.mean(clean[name])))
        per_src_last = {n: {s: round(float(np.mean(v)), 3) for s, v in d.items()}
                        for n, d in psrc.items()}
        print(f"  seed {seed}: " + "  ".join(
            f"{n}={np.mean(ious[n]):.3f}" for n in segfns if ious[n]))

    print(f"\n{'='*78}")
    print(f"{'segmenter':<12}{'crack mIoU (±std)':>22}{'noncrack 클린율':>18}{'s/img':>8}")
    print("-" * 78)
    summary = {}
    for name in segfns:
        if not crack_miou[name]:
            continue
        m, sd = float(np.mean(crack_miou[name])), float(np.std(crack_miou[name]))
        fc = float(np.mean(fp_clean[name])) if fp_clean[name] else None
        summary[name] = {"crack_mIoU_mean": round(m, 4), "crack_mIoU_std": round(sd, 4),
                         "noncrack_clean_rate": round(fc, 3) if fc is not None else None,
                         "per_source_lastseed": per_src_last.get(name, {}),
                         "sec_per_img": round(timing[name] / max(count[name], 1), 3)}
        print(f"{name:<12}{m:>14.3f} ±{sd:.3f}{(f'{fc:>16.2f}' if fc is not None else f'{chr(45):>16}')}"
              f"{timing[name]/max(count[name],1):>8.2f}")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "gauge_bench.json", "w") as f:
        json.dump({"n_per_seed": args.n, "seeds": args.seeds,
                   "commercial_only": args.commercial_only, "results": summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: experiments/results/gauge_bench.json")
    print("주: crack mIoU는 빈GT 제외(분할 능력만). noncrack 클린율은 탐지 능력(거짓양성 회피) 별도 보고.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
