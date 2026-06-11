"""E-cnt-2 — SAHI-style tiled SAM3 inference for rebar counting.

This tests whether the E-cnt-1 failure is mainly a scale/crowding issue. The
script runs SAM3 on overlapping tiles, maps tile masks back to the full image,
deduplicates instances, and compares counts to ROI-1555 labelme instance counts.

It is intentionally an inference-only intervention; no fine-tuning or labels are
used except for evaluation.

Usage:
    python experiments/rebar_sahi_eval.py --n 10 --prompt "metal rod" --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.rebar_count_eval import load_pairs  # noqa: E402
from gaugeanything.segmenters import Instance, segment_sam3  # noqa: E402


def tile_windows(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    if not xs or xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    if not ys or ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    out = []
    for y0 in ys:
        for x0 in xs:
            out.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def center(inst: Instance) -> tuple[float, float]:
    y1, x1 = np.nonzero(inst.mask)
    return float(x1.mean()), float(y1.mean())


def dedup(insts: list[Instance], iou_thr: float = 0.35, center_thr_px: float = 18.0) -> list[Instance]:
    insts = sorted(insts, key=lambda i: (-i.score, -int(i.mask.sum())))
    kept: list[Instance] = []
    for inst in insts:
        cx, cy = center(inst)
        duplicate = False
        for k in kept:
            kx, ky = center(k)
            if abs(cx - kx) <= center_thr_px and abs(cy - ky) <= center_thr_px:
                duplicate = True
                break
            if mask_iou(inst.mask, k.mask) >= iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def sahi_segment(image: np.ndarray, prompt: str, tile: int, overlap: float,
                 threshold: float, min_area: int) -> list[Instance]:
    h, w = image.shape[:2]
    full: list[Instance] = []
    for x0, y0, x1, y1 in tile_windows(h, w, tile, overlap):
        crop = image[y0:y1, x0:x1]
        try:
            insts = segment_sam3(crop, prompt, threshold=threshold)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  ! tile failure {x0},{y0},{x1},{y1}: {str(e)[:120]}")
            continue
        for inst in insts:
            if inst.mask.sum() < min_area:
                continue
            m = np.zeros((h, w), dtype=bool)
            m[y0:y1, x0:x1] = inst.mask
            full.append(Instance(mask=m, score=inst.score, label=prompt))
    return dedup(full)


def summarize(rows: list[dict]) -> dict:
    errs = np.array([abs(r["pred"] - r["gt"]) for r in rows], dtype=float)
    rels = np.array([abs(r["pred"] - r["gt"]) / max(r["gt"], 1) for r in rows], dtype=float)
    return {
        "n": len(rows),
        "MAE": round(float(errs.mean()), 2),
        "rel_err_mean": round(float(rels.mean()), 4),
        "count_acc@10pct": round(float((rels <= 0.10).mean()), 3),
        "exact_rate": round(float((errs == 0).mean()), 3),
        "gt_mean": round(float(np.mean([r["gt"] for r in rows])), 1),
        "pred_mean": round(float(np.mean([r["pred"] for r in rows])), 1),
    }


def make_gallery(image: np.ndarray, insts: list[Instance], gt: int, pred: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(image)
    ax[0].set_title(f"Input (GT count = {gt})")
    ax[0].axis("off")
    ax[1].imshow(image)
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    rng = np.random.default_rng(3)
    for inst in insts:
        c = rng.random(3) * 0.75 + 0.25
        overlay[inst.mask] = [*c, 0.52]
    ax[1].imshow(overlay)
    ax[1].set_title(f"SAHI-SAM3 tiled count = {pred}")
    ax[1].axis("off")
    fig.suptitle("E-cnt-2: tiled SAM3 rebar counting", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="metal rod")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()

    from PIL import Image

    pairs = load_pairs(args.n, seed=args.seed)
    rows = []
    gallery_done = False
    print("=== E-cnt-2: SAHI-style tiled SAM3 rebar counting ===")
    print(f"n={len(pairs)} prompt={args.prompt!r} tile={args.tile} overlap={args.overlap} threshold={args.threshold}")
    for k, (ip, gt) in enumerate(pairs, start=1):
        image = np.array(Image.open(ip).convert("RGB"))
        insts = sahi_segment(image, args.prompt, args.tile, args.overlap, args.threshold, args.min_area)
        pred = len(insts)
        rows.append({"img": ip.name, "gt": gt, "pred": pred, "abs_err": abs(pred - gt)})
        print(f"  {k:02d}/{len(pairs)} {ip.name}: GT={gt:3d} pred={pred:3d} err={abs(pred-gt):3d}")
        if args.gallery and not gallery_done and gt >= 15:
            make_gallery(image, insts, gt, pred, Path(args.gallery) / "rebar_sahi_count.png")
            gallery_done = True

    summary = summarize(rows)
    out = {
        "config": {
            "prompt": args.prompt,
            "n": args.n,
            "seed": args.seed,
            "tile": args.tile,
            "overlap": args.overlap,
            "threshold": args.threshold,
            "min_area": args.min_area,
            "note": "SAHI-style tiled SAM3 inference; no training.",
        },
        "summary": summary,
        "rows": rows,
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/rebar_sahi_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nsummary:", summary)
    print("결과 저장: experiments/results/rebar_sahi_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
