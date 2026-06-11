"""E-cnt-1 — rebar 카운팅 MAE (로드맵 2단계: 카운팅 능력 실증).

데이터: ROI-1555 (labelme JSON, 폴리곤 인스턴스 — GT count = shapes 수).
프로토콜: SAM3 zero-shot 개념 프롬프트 → 인스턴스 수 vs GT → MAE·상대오차·정확일치율.
프롬프트 민감도 교훈 반영: 2개 프롬프트 보고 (best 명시).
참고 앵커: UAV rebar count acc 86.27% (지도학습, Sci Rep 2025) — 우리는 zero-shot.

실행: python experiments/rebar_count_eval.py --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaugeanything.segmenters import segment_sam3  # noqa: E402

ROOT = Path("datasets/rebar_roi1555")


def load_pairs(n, seed=0):
    """(이미지, GT count) 쌍. labelme json과 같은 폴더의 jpg."""
    jsons = sorted(ROOT.glob("*/img_label/*.json"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(jsons))[: n * 2]
    out = []
    for i in idx:
        jp = jsons[i]
        d = json.loads(jp.read_text())
        ip = jp.with_suffix(".jpg")
        if not ip.exists():
            cand = jp.parent / Path(d.get("imagePath", "")).name
            if cand.exists():
                ip = cand
            else:
                continue
        out.append((ip, len(d.get("shapes", []))))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--prompts", nargs="+", default=["rebar", "steel bar"])
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()

    from PIL import Image
    pairs = load_pairs(args.n)
    print(f"=== E-cnt-1: rebar 카운팅 (ROI-1555, n={len(pairs)}) ===")
    results = {}
    for prompt in args.prompts:
        errs, rels, exact, rows = [], [], 0, []
        first = True
        for ip, gt in pairs:
            img = np.array(Image.open(ip).convert("RGB"))
            insts = segment_sam3(img, prompt, threshold=0.4)
            # 소형 노이즈 제거
            insts = [i for i in insts if i.mask.sum() >= 100]
            pred = len(insts)
            errs.append(abs(pred - gt)); rels.append(abs(pred - gt) / max(gt, 1))
            exact += int(pred == gt)
            rows.append({"img": ip.name, "gt": gt, "pred": pred})
            if args.gallery and first and gt >= 15 and abs(pred - gt) <= max(2, gt * 0.1):
                _panel(img, insts, gt, pred, prompt, Path(args.gallery) / "rebar_count.png")
                first = False
        r = {"n": len(pairs), "MAE": round(float(np.mean(errs)), 2),
             "rel_err_mean": round(float(np.mean(rels)), 4),
             "count_acc@10pct": round(float(np.mean([e <= 0.10 for e in rels])), 3),
             "exact_rate": round(exact / len(pairs), 3),
             "gt_mean": round(float(np.mean([g for _, g in pairs])), 1)}
        results[prompt] = {"summary": r, "rows": rows}
        print(f"\n[prompt='{prompt}'] MAE {r['MAE']} (GT평균 {r['gt_mean']}) · "
              f"상대오차 {r['rel_err_mean']*100:.1f}% · ±10% 정확률 {r['count_acc@10pct']*100:.0f}% · "
              f"정확일치 {r['exact_rate']*100:.0f}%")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "rebar_count_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n결과 저장: experiments/results/rebar_count_eval.json")
    print("주: zero-shot. 지도학습 앵커(UAV rebar) count acc 86.27%. 밀집 한계는 SAM3 쿼리 슬롯.")
    return 0


def _panel(img, insts, gt, pred, prompt, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(img); ax[0].set_title(f"Input (GT count = {gt})"); ax[0].axis("off")
    ax[1].imshow(img)
    overlay = np.zeros((*img.shape[:2], 4), np.float32)
    rng = np.random.default_rng(0)
    for inst in insts:
        c = rng.random(3) * 0.8 + 0.2
        overlay[inst.mask] = [*c, 0.55]
    ax[1].imshow(overlay)
    ax[1].set_title(f"SAM3 '{prompt}' → count = {pred}"); ax[1].axis("off")
    fig.suptitle("E-cnt-1: zero-shot rebar counting", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94]); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
