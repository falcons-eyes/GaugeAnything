"""멀티도메인 일반화 벤치 — GaugeAnything이 크랙 너머로 일반화하는가.

같은 파이프라인(SAM3 + 측정), 다른 프롬프트, 다른 도메인/측정원시:
  - 콘크리트 크랙 (CrackSeg9k)        → thin → width
  - 자성타일 결함 5종 (Magnetic tile)  → blowhole=blob/직경, crack=thin/width, ...
측정원시는 geometry.classify_kind가 형태로 자동 선택 (thin vs blob).

각 도메인: SAM3 mIoU(vs GT 마스크) + 측정(auto) + 정성 갤러리.
실행: python experiments/gauge_multidomain.py --per 30 --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_bench import find_pairs as crack_pairs  # noqa: E402
from experiments.gauge_bench import iou_f1  # noqa: E402
from gaugeanything.geometry import classify_kind, measure  # noqa: E402
from gaugeanything.segmenters import segment_sam3  # noqa: E402

DS = "datasets"


def mt_pairs(defect: str) -> list[tuple[Path, Path, str]]:
    """자성타일: MT_<defect>/Imgs/*.jpg + 동일 stem *.png(GT 마스크)."""
    d = Path(DS) / "magnetic_tile" / f"MT_{defect}" / "Imgs"
    out = []
    for jpg in sorted(d.glob("*.jpg")):
        png = jpg.with_suffix(".png")
        if png.exists():
            out.append((jpg, png, f"mt_{defect.lower()}"))
    return out


def load_pair(img_p: Path, mask_p: Path):
    from PIL import Image
    img = np.array(Image.open(img_p).convert("RGB"))
    gt = np.array(Image.open(mask_p).convert("L")) > 127
    return img, gt


# (도메인명, 페어 샘플러, SAM3 프롬프트, 표시명[ASCII — matplotlib 폰트])
DOMAINS = [
    ("crack_concrete", lambda: crack_pairs(Path(DS) / "crackseg9k"), "crack", "Concrete crack"),
    ("mt_blowhole", lambda: mt_pairs("Blowhole"), "hole", "Mag-tile blowhole"),
    ("mt_crack", lambda: mt_pairs("Crack"), "crack", "Mag-tile crack"),
    ("mt_break", lambda: mt_pairs("Break"), "crack", "Mag-tile break"),
    ("mt_fray", lambda: mt_pairs("Fray"), "scratch", "Mag-tile fray"),
    ("mt_uneven", lambda: mt_pairs("Uneven"), "stain", "Mag-tile uneven"),
]


def render_panel(img, mask, gt, stats, kind, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from skimage.morphology import skeletonize

    gray = img if img.ndim == 2 else img.mean(2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input"); ax[0].axis("off")

    ov = np.stack([gray]*3, -1).astype(np.float32)/255 if gray.max() > 1 else np.stack([gray]*3,-1)
    ov[mask] = ov[mask]*0.4 + np.array([1, 0, 0])*0.6
    ax[1].imshow(ov); ax[1].set_title(f"SAM3 ('{title[1]}')  IoU={title[2]:.2f}"); ax[1].axis("off")

    ax[2].imshow(gray, cmap="gray")
    if kind == "thin" and mask.sum() > 20:
        edt = ndimage.distance_transform_edt(mask); skel = skeletonize(mask)
        heat = ndimage.grey_dilation(np.where(skel, 2*edt, 0), size=3)
        hm = np.ma.masked_where(heat == 0, heat)
        im = ax[2].imshow(hm, cmap="jet"); plt.colorbar(im, ax=ax[2], fraction=.046, label="width (px)")
        ax[2].set_title(f"Width profile · mean {stats['mean']:.1f}px")
    else:  # blob
        ax[2].contour(mask, colors="lime", linewidths=1.2)
        ax[2].set_title(f"Blob · equiv-dia {stats['dia']:.1f}px, area {int(stats['area'])}px")
    ax[2].axis("off")
    fig.suptitle(f"GaugeAnything · {title[0]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30, help="도메인당 평가 이미지 수")
    ap.add_argument("--gallery", default=None, help="갤러리 출력 디렉토리")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=== 멀티도메인 일반화 벤치 (SAM3 + 측정) ===")
    summary = {}
    for dom, sampler, prompt, label in DOMAINS:
        pairs = sampler()
        if not pairs:
            print(f"  ! {dom}: 데이터 없음"); continue
        # 크랙은 noncrack/소스 균형 무시하고 무작위, GT 있는 것만
        idx = rng.permutation(len(pairs))[:args.per]
        sample = [pairs[i] for i in idx]
        ious, dias, widths, kinds = [], [], [], []
        first_gallery = True
        t0 = time.time()
        for j, (ip, mp, _) in enumerate(sample):
            img, gt = load_pair(ip, mp)
            if gt.sum() < 30:
                continue
            try:
                insts = segment_sam3(img, prompt, threshold=0.4)
            except Exception as e:
                print(f"  ! {dom} SAM3 실패: {str(e)[:90]}"); break
            mask = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(gt.shape, bool)
            iou, _ = iou_f1(mask, gt)
            ious.append(iou)
            if mask.sum() > 20:
                k = classify_kind(mask)
                g = measure(mask, kind=k)
                kinds.append(k)
                if k == "blob":
                    dias.append(g.equiv_diameter)
                else:
                    widths.append(g.width_mean)
                if args.gallery and first_gallery and iou > 0.2:
                    stats = {"mean": g.width_mean, "dia": g.equiv_diameter, "area": g.area}
                    render_panel(img, mask, gt, stats, k, Path(args.gallery)/f"dom_{dom}.png",
                                 (label, prompt, iou))
                    first_gallery = False
        from collections import Counter
        s = {"n": len(ious), "mIoU": round(float(np.mean(ious)), 4) if ious else 0,
             "kind": dict(Counter(kinds)),
             "mean_dia_px": round(float(np.mean(dias)), 1) if dias else None,
             "mean_width_px": round(float(np.mean(widths)), 1) if widths else None,
             "sec_per_img": round((time.time()-t0)/max(len(ious), 1), 2)}
        summary[dom] = s
        meas = f"dia {s['mean_dia_px']}px" if s["mean_dia_px"] else f"width {s['mean_width_px']}px"
        print(f"  {label:22s} prompt='{prompt}'  mIoU={s['mIoU']:.3f}  ({s['kind']}, {meas}, n={s['n']})")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out/"gauge_multidomain.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n결과 저장: experiments/results/gauge_multidomain.json")
    if summary:
        mi = np.mean([s["mIoU"] for s in summary.values()])
        print(f"도메인 평균 mIoU: {mi:.3f} · 측정원시 자동선택(thin/blob)으로 도메인 가로지름")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
