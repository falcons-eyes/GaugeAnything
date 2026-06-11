"""Soft Inspection 검증 — binary가 실패한 결함을 연속 표현이 잡는가.

가설:
  uneven(경계없는 밝기장) → 조명잔차 soft맵이 SAM3 binary보다 결함을 잘 랭크 (ROC-AUC).
  fray(애매경계) → guided-matte로 경계 soft화 (정성).

지표: 픽셀 ROC-AUC (연속맵 vs GT 마스크) — 임계 없이 "결함>정상" 랭킹 측정.
실행: python experiments/soft_inspection_eval.py --per 25 --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_multidomain import load_pair, mt_pairs  # noqa: E402
from gaugeanything.soft import guided_matte, illumination_residual, mura_severity, severity_score  # noqa: E402


def auc(score: np.ndarray, gt: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    y = gt.ravel().astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        return None
    s = score.ravel().astype(np.float32)
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=25)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--with-sam3", action="store_true", help="SAM3 binary AUC 비교(느림)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sam3 = None
    if args.with_sam3:
        from gaugeanything.segmenters import segment_sam3
        sam3 = segment_sam3

    print("=== Soft Inspection 검증 (조명잔차 vs binary) ===")
    out = {}
    for defect, prompt in [("Uneven", "stain"), ("Fray", "scratch")]:
        pairs = mt_pairs(defect)
        if not pairs:
            print(f"  ! MT_{defect} 없음"); continue
        idx = rng.permutation(len(pairs))[:args.per]
        res_auc, raw_auc, sam_auc, sas, sqs = [], [], [], [], []
        gallery_done = False
        for k in idx:
            ip, mp, _ = pairs[k]
            img, gt = load_pair(ip, mp)
            if gt.sum() < 30:
                continue
            # 조명잔차 soft맵
            sev = mura_severity(img, order=2)
            soft = sev["soft_map"]
            a = auc(soft, gt)
            if a is not None:
                res_auc.append(a); sas.append(sev["Sa"]); sqs.append(sev["Sq"])
            # raw grayscale 편차 (베이스라인)
            g = img.mean(2) if img.ndim == 3 else img
            raw = np.abs(g - g.mean())
            ar = auc(raw, gt)
            if ar is not None:
                raw_auc.append(ar)
            # SAM3 binary (선택)
            if sam3 is not None:
                try:
                    insts = sam3(img, prompt, threshold=0.4)
                    m = np.any([i.mask for i in insts], axis=0) if insts else np.zeros(gt.shape, bool)
                    asm = auc(m.astype(float), gt)
                    if asm is not None:
                        sam_auc.append(asm)
                except Exception:
                    pass
            # 갤러리 (1장)
            if args.gallery and not gallery_done and a and a > 0.6:
                _panel(img, gt, soft, sev, defect, Path(args.gallery)/f"soft_{defect.lower()}.png")
                gallery_done = True
        sgrade = severity_score(mura_severity(load_pair(*pairs[idx[0]][:2])[0])["soft_map"])
        out[defect] = {
            "n": len(res_auc),
            "residual_AUC": round(float(np.mean(res_auc)), 4) if res_auc else None,
            "raw_gray_AUC": round(float(np.mean(raw_auc)), 4) if raw_auc else None,
            "sam3_binary_AUC": round(float(np.mean(sam_auc)), 4) if sam_auc else None,
            "Sa": round(float(np.mean(sas)), 3) if sas else None,
            "Sq": round(float(np.mean(sqs)), 3) if sqs else None,
        }
        o = out[defect]
        print(f"\n[MT_{defect}] prompt='{prompt}'  n={o['n']}")
        print(f"  조명잔차 soft AUC : {o['residual_AUC']}   (raw-gray {o['raw_gray_AUC']}"
              + (f", SAM3-binary {o['sam3_binary_AUC']}" if sam3 else "") + ")")
        print(f"  severity Sa={o['Sa']} Sq={o['Sq']}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/soft_inspection.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n결과 저장: experiments/results/soft_inspection.json")
    print("주: AUC>0.5면 연속맵이 결함을 정상보다 높게 랭크 — 임계 없이도 작동(binary IoU≈0과 대조)")
    return 0


def _panel(img, gt, soft, sev, defect, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gray = img.mean(2) if img.ndim == 3 else img
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT mask"); ax[1].axis("off")
    im = ax[2].imshow(soft, cmap="inferno"); ax[2].set_title("Illumination-residual soft map")
    plt.colorbar(im, ax=ax[2], fraction=.046); ax[2].axis("off")
    ax[3].imshow(gray, cmap="gray"); ax[3].imshow(soft, cmap="inferno", alpha=0.55)
    ax[3].set_title(f"Overlay · Sa={sev['Sa']:.2f} Sq={sev['Sq']:.2f}"); ax[3].axis("off")
    fig.suptitle(f"Soft Inspection · MT_{defect} (field-modeling, no boundary)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
