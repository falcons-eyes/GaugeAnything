"""E-mm-1 — 동전 cross-coin mm 검증 (현장 없는 실사진 계측 검증).

데이터: kaa/coins-dataset src/<권종>/ — 동일 권종 동전들이 한 scene에 (테이블 위, 마커 없음).
유로 권종별 직경은 법정 고정 → **leave-one-out**: 동전 i의 직경을, 나머지 동전들의
픽셀 직경 평균을 기지 치수(법정 직경)로 삼아 환산 → 법정 직경과 비교.

검증 범위(정직): 절대 스케일 체인(마커)이 아니라 **분할→직경 측정 체인의 일관성**
(SAM3 분할 + 등가직경 + known-object 리졸버 경로). 마커 체인은 합성 검증(0.38%) 완료.

실행: python experiments/coins_mm_eval.py --per-denom 4 --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaugeanything.segmenters import segment_sam3  # noqa: E402

# 유로 법정 직경 (mm)
EURO_MM = {"1c": 16.25, "2c": 18.75, "5c": 21.25, "10c": 19.75,
           "20c": 22.25, "50c": 24.25, "1e": 23.25, "2e": 25.75}
SRC = Path("datasets/coins/kaa/src")


def coin_diameters_px(img: np.ndarray, min_area: int = 2000, circ_min: float = 0.72):
    """SAM3 'coin' → 원형 인스턴스 필터 → 등가직경(px) 리스트 + 마스크."""
    from scipy import ndimage
    insts = segment_sam3(img, "coin", threshold=0.4)
    out = []
    for inst in insts:
        m = inst.mask
        a = float(m.sum())
        if a < min_area:
            continue
        # 둘레: 침식 차분 근사
        er = ndimage.binary_erosion(m)
        per = float((m & ~er).sum()) or 1.0
        circ = 4 * np.pi * a / (per ** 2)
        if circ < circ_min:
            continue
        out.append((2.0 * np.sqrt(a / np.pi), m, inst.score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-denom", type=int, default=4)
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()

    from PIL import Image
    rows, rel_errs, cvs = [], [], []
    first_panel = True
    print("=== E-mm-1: 동전 cross-coin mm 검증 (leave-one-out known-object) ===")
    for denom, D in sorted(EURO_MM.items()):
        d = SRC / denom
        if not d.exists():
            continue
        files = sorted([*d.glob("*.jpg"), *d.glob("*.JPG")])[: args.per_denom]
        for fp in files:
            img = np.array(Image.open(fp).convert("RGB"))
            coins = coin_diameters_px(img)
            if len(coins) < 3:
                print(f"  ! {denom}/{fp.name}: 동전 {len(coins)}개 (<3) — 스킵")
                continue
            dpx = np.array([c[0] for c in coins])
            # leave-one-out: 각 동전을 나머지 평균으로 환산
            errs_i = []
            for i in range(len(dpx)):
                ref = np.delete(dpx, i).mean()
                pred = dpx[i] * D / ref
                errs_i.append(abs(pred - D) / D)
            rel = float(np.mean(errs_i))
            cv = float(dpx.std() / dpx.mean())
            rel_errs.append(rel); cvs.append(cv)
            rows.append({"denom": denom, "file": fp.name, "n_coins": len(dpx),
                         "rel_err": round(rel, 4), "cv_px": round(cv, 4)})
            print(f"  {denom:>3} {fp.name:16s} 동전 {len(dpx):2d}개  "
                  f"LOO 상대오차 {rel*100:5.2f}%  px-CV {cv*100:4.2f}%")
            if args.gallery and first_panel and len(dpx) >= 5:
                _panel(img, coins, D, Path(args.gallery) / "coins_mm.png", denom)
                first_panel = False

    if not rel_errs:
        print("평가 샘플 없음"); return 1
    rel = np.array(rel_errs)
    summary = {"n_images": len(rows),
               "rel_err_mean": round(float(rel.mean()), 4),
               "rel_err_median": round(float(np.median(rel)), 4),
               "cv_mean": round(float(np.mean(cvs)), 4),
               "pass@5pct": round(float((rel <= 0.05).mean()), 3),
               "pass@10pct": round(float((rel <= 0.10).mean()), 3),
               "note": "LOO 일관성 검증 — 분할+직경 체인. 절대 마커 체인은 합성 0.38% 별도"}
    print(f"\n{'='*60}\nE-mm-1 (n={summary['n_images']} imgs): "
          f"상대오차 평균 {summary['rel_err_mean']*100:.2f}% / 중앙값 {summary['rel_err_median']*100:.2f}% "
          f"· ±5% 합격 {summary['pass@5pct']*100:.0f}% · ±10% {summary['pass@10pct']*100:.0f}%")
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "coins_mm_eval.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/coins_mm_eval.json")
    return 0


def _panel(img, coins, D, out_path, denom):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].imshow(img); ax[0].set_title(f"Input ({denom}, legal dia {D}mm)"); ax[0].axis("off")
    ax[1].imshow(img)
    dpx = np.array([c[0] for c in coins])
    for dp, m, s in coins:
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean(), xs.mean()
        ref = (dpx.sum() - dp) / (len(dpx) - 1)
        pred = dp * D / ref
        ax[1].add_patch(plt.Circle((cx, cy), dp / 2, fill=False, color="lime", lw=2))
        ax[1].text(cx, cy - dp / 2 - 8, f"{pred:.1f}mm", color="lime",
                   ha="center", fontsize=10, weight="bold")
    ax[1].set_title("SAM3 'coin' → equiv-dia → LOO known-object mm"); ax[1].axis("off")
    fig.suptitle("E-mm-1: real-photo measurement consistency (no marker, no field rig)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94]); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
