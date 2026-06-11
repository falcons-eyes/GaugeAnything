"""GaugeAnything end-to-end 시각 데모.

실제 크랙 이미지 → SAM3 분할 → 폭 프로파일 측정 → 주석 이미지 출력.
프로젝트 페이지용 아티팩트 + 파이프라인 실동작 시각 확인.

산출: 원본 | SAM3 마스크 | 폭 히트맵(스켈레톤) 3패널 + 측정 readout.
mm GT 부재 시 --mm-per-px 가정값으로 mm 표기 (없으면 px).

실행: python experiments/gauge_demo.py --auto-pick crack500 --mm-per-px 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaugeanything.geometry import measure_thin  # noqa: E402
from gaugeanything.segmenters import segment_sam3, segment_threshold  # noqa: E402


def pick_image(root: Path, source: str) -> Path:
    """CrackSeg9k에서 해당 소스의 크랙 이미지 1장 (마스크 큰 것 우선)."""
    from PIL import Image
    masks = {m.stem: m for m in root.rglob("Final_Masks/Masks/*.png")}
    cands = []
    for img in root.rglob("Images*/*"):
        if img.suffix.lower() not in (".jpg", ".png") or img.stem not in masks:
            continue
        if not img.stem.lower().startswith(source.lower()):
            continue
        cands.append((img, masks[img.stem]))
        if len(cands) >= 40:
            break
    # 크랙 면적 큰 이미지 선택 (시각적으로 명확)
    best = max(cands, key=lambda p: (np.array(Image.open(p[1]).convert("L")) > 127).sum())
    return best[0]


def width_heatmap(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """스켈레톤 위 국소 폭(2×EDT) 히트맵 + 통계."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    m = mask.astype(bool)
    edt = ndimage.distance_transform_edt(m)
    skel = skeletonize(m)
    widths = 2.0 * edt[skel]
    heat = np.zeros(mask.shape, np.float32)
    heat[skel] = 2.0 * edt[skel]
    # 시각화 위해 스켈레톤 두껍게
    heat = ndimage.grey_dilation(heat, size=3)
    g = measure_thin(m)
    stats = {"mean": g.width_mean, "max": g.width_max, "p95": g.width_p95, "len": g.length}
    return heat, stats


def render(img: np.ndarray, mask: np.ndarray, heat: np.ndarray, stats: dict,
           mm_per_px: float | None, out_path: Path, prompt: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "mm" if mm_per_px else "px"
    s = mm_per_px if mm_per_px else 1.0
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    gray = img if img.ndim == 2 else img.mean(2)

    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input (real crack)"); ax[0].axis("off")

    ov = np.stack([gray] * 3, -1).astype(np.float32) / 255 if gray.max() > 1 else np.stack([gray]*3,-1)
    ov[mask] = ov[mask] * 0.4 + np.array([1, 0, 0]) * 0.6
    ax[1].imshow(ov); ax[1].set_title(f"SAM3 segmentation ('{prompt}')  mask px={int(mask.sum())}"); ax[1].axis("off")

    hm = np.ma.masked_where(heat == 0, heat * s)
    ax[2].imshow(gray, cmap="gray")
    im = ax[2].imshow(hm, cmap="jet")
    plt.colorbar(im, ax=ax[2], fraction=0.046, label=f"local width ({unit})")
    ax[2].set_title("Width profile (skeleton)"); ax[2].axis("off")

    txt = (f"mean width {stats['mean']*s:.2f}{unit} | max {stats['max']*s:.2f} | "
           f"p95 {stats['p95']*s:.2f} | length {stats['len']*s:.1f}{unit}"
           + (f"  (assumed {mm_per_px} mm/px)" if mm_per_px else "  (no scale -> px)"))
    fig.suptitle("GaugeAnything - promptable quantitative inspection:  " + txt, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"저장: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--root", default="datasets/crackseg9k")
    ap.add_argument("--auto-pick", default="crack500", help="CrackSeg9k 소스 접두사")
    ap.add_argument("--segmenter", default="sam3", choices=["sam3", "threshold"])
    ap.add_argument("--prompt", default="crack")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--out", default="experiments/results/gauge_demo.png")
    args = ap.parse_args()

    from PIL import Image
    img_path = Path(args.image) if args.image else pick_image(Path(args.root), args.auto_pick)
    print(f"이미지: {img_path}")
    img = np.array(Image.open(img_path).convert("RGB"))

    if args.segmenter == "sam3":
        insts = segment_sam3(img, args.prompt, threshold=0.4)
    else:
        insts = segment_threshold(img, args.prompt)
    print(f"인스턴스: {len(insts)}")
    if not insts:
        print("탐지 없음 — threshold 폴백")
        insts = segment_threshold(img, args.prompt)
    mask = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(img.shape[:2], bool)

    heat, stats = width_heatmap(mask)
    print(f"측정: 평균폭 {stats['mean']:.2f}px, 최대 {stats['max']:.2f}px, 길이 {stats['len']:.1f}px")
    render(img, mask, heat, stats, args.mm_per_px, Path(args.out), args.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
