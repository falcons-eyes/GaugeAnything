"""E-mm-2b — T-LESS에 SAM3 실분할: promptable 산업부품 계측의 실제 성능.

N2(완벽 마스크 상한: 중앙값 2.83%)와 동일 프로토콜·동일 케이스에서 마스크만
SAM3 zero-shot으로 교체 → 갭이 곧 "분할이 만드는 측정 비용".

분리 보고:
  ① 매칭률 (GT mask_visib와 IoU≥0.3인 SAM3 인스턴스 존재) — 분할/개념 실패
  ② 매칭된 케이스의 마스크 IoU — 분할 품질
  ③ 매칭된 케이스의 chord 측정 rel_err vs CAD GT — 측정 열화 (N2 앵커 2.83%)

위험 요인(감사 명시): T-LESS는 텍스처리스 전기부품 — SAM3 개념 프롬프트가
자연스럽지 않은 도메인. 실패해도 정직 보고 가치.

실행: python experiments/tless_sam3_eval.py --scenes 6 --imgs 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.tless_upper_bound import TLESS, load_ply_vertices, max_chord_2d  # noqa: E402
from gaugeanything.segmenters import segment_sam3  # noqa: E402

PROMPTS = ["electrical component", "plastic part", "white object", "industrial part"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--imgs", type=int, default=3)
    ap.add_argument("--visib", type=float, default=0.95)
    ap.add_argument("--iou-match", type=float, default=0.3)
    args = ap.parse_args()

    from PIL import Image
    models_dir = TLESS / "models_eval"
    scene_dirs = sorted((TLESS / "test_primesense").iterdir())[: args.scenes]
    print(f"=== E-mm-2b T-LESS × SAM3 (prompts={PROMPTS}) ===")

    model_cache: dict[int, np.ndarray] = {}
    # per-prompt 집계
    agg = {p: {"n_gt": 0, "matched": 0, "ious": [], "rels": []} for p in PROMPTS}

    for sd in scene_dirs:
        cam = json.loads((sd / "scene_camera.json").read_text())
        gt = json.loads((sd / "scene_gt.json").read_text())
        gti = json.loads((sd / "scene_gt_info.json").read_text())
        im_ids = sorted(gt.keys(), key=int)[: args.imgs]
        for im in im_ids:
            rgb_p = sd / "rgb" / f"{int(im):06d}.png"
            if not rgb_p.exists():
                rgb_p = sd / "rgb" / f"{int(im):06d}.jpg"
            img = np.array(Image.open(rgb_p).convert("RGB"))
            K = np.array(cam[im]["cam_K"]).reshape(3, 3)
            fx = K[0, 0]

            # GT 객체 목록 (visib 필터) + GT chord mm
            cases = []
            for gi, (obj, info) in enumerate(zip(gt[im], gti[im])):
                if info.get("visib_fract", 0) < args.visib:
                    continue
                oid = obj["obj_id"]
                if oid not in model_cache:
                    model_cache[oid] = load_ply_vertices(models_dir / f"obj_{oid:06d}.ply")
                R = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj["cam_t_m2c"]).reshape(3)
                Xc = model_cache[oid] @ R.T + t
                P = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2],
                              K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
                _, i, j = max_chord_2d(P)
                gt_mm = float(np.linalg.norm(Xc[i] - Xc[j]))
                mp = sd / "mask_visib" / f"{int(im):06d}_{gi:06d}.png"
                if not mp.exists():
                    continue
                gmask = np.array(Image.open(mp)) > 0
                if gmask.sum() < 30:
                    continue
                cases.append({"oid": oid, "gt_mm": gt_mm, "gmask": gmask, "tz": float(t[2])})
            if not cases:
                continue

            for prompt in PROMPTS:
                insts = segment_sam3(img, prompt, threshold=0.3)
                a = agg[prompt]
                for c in cases:
                    a["n_gt"] += 1
                    best_iou, best_m = 0.0, None
                    for inst in insts:
                        inter = float((inst.mask & c["gmask"]).sum())
                        if inter == 0:
                            continue
                        iou = inter / float((inst.mask | c["gmask"]).sum())
                        if iou > best_iou:
                            best_iou, best_m = iou, inst.mask
                    if best_iou < args.iou_match:
                        continue
                    a["matched"] += 1
                    a["ious"].append(best_iou)
                    ys, xs = np.nonzero(best_m)
                    chord, _, _ = max_chord_2d(np.stack([xs, ys], 1).astype(np.float64))
                    pred_mm = chord * c["tz"] / fx
                    a["rels"].append(abs(pred_mm - c["gt_mm"]) / max(c["gt_mm"], 1e-6))

    out = {}
    print(f"\n{'prompt':<24}{'match':>8}{'IoU중앙':>9}{'rel중앙':>9}{'±10%':>7}")
    print("-" * 60)
    for p, a in agg.items():
        if a["n_gt"] == 0:
            continue
        mr = a["matched"] / a["n_gt"]
        med_iou = float(np.median(a["ious"])) if a["ious"] else 0.0
        rels = np.array(a["rels"]) if a["rels"] else np.array([np.nan])
        out[p] = {"n_gt": a["n_gt"], "match_rate": round(mr, 3),
                  "iou_median": round(med_iou, 3),
                  "rel_err_median": round(float(np.nanmedian(rels)), 4),
                  "pass@10pct": round(float(np.nanmean(rels <= 0.10)), 3)}
        print(f"{p:<24}{mr*100:>7.0f}%{med_iou:>9.3f}"
              f"{float(np.nanmedian(rels))*100:>8.1f}%{float(np.nanmean(rels<=0.10))*100:>6.0f}%")
    print("\nN2 앵커(완벽 마스크): rel중앙 2.83%, ±10% 94%")
    res = Path("experiments/results"); res.mkdir(parents=True, exist_ok=True)
    (res / "tless_sam3_eval.json").write_text(json.dumps(
        {"prompts": out, "anchor_N2": {"rel_err_median": 0.0283, "pass@10pct": 0.94},
         "protocol": "N2 동일 케이스, mask=SAM3, IoU>=0.3 매칭"}, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/tless_sam3_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
