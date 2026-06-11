"""N2/E-mm-2 — T-LESS 측정 상한: 완벽한 마스크 + plane-scale로 산업부품 치수는 몇 %인가.

분리 질문(RESEARCH_AUDIT N2): 분할이 완벽하다면(GT mask_visib 사용), 우리의
단일-깊이 plane-scale 변환(mm_per_px = Z/fx)이 3D 산업부품에서 얼마나 정확한가?
— foreshortening·깊이변화가 만드는 **방법론적 상한**을 정량화. SAM3 난이도와 분리.

GT 유도 (BOP, 전부 mm):
  모델 정점 V → 포즈(R,t)로 카메라계 Xc → K로 투영 P(px).
  투영 최대 chord 쌍 (i,j) → GT mm chord = ||Xc_i − Xc_j||.
예측: GT visible mask의 최대 chord(px) × (t_z / fx).
필터: visib_fract ≥ 0.95 (가림 없는 케이스만 — 상한 측정이므로).

실행: python experiments/tless_upper_bound.py --scenes 6 --imgs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TLESS = Path("datasets/tless")


def load_ply_vertices(path: Path, max_v: int = 3000) -> np.ndarray:
    """PLY 정점 (x,y,z) 로드 (ASCII/binary_little_endian) + 서브샘플."""
    with open(path, "rb") as f:
        n_v, fmt, vprops, in_vertex = 0, "ascii", [], False
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_v = int(line.split()[-1]); in_vertex = True
            elif line.startswith("element"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                vprops.append(line.split()[1])  # 타입 (float, uchar, ...)
            elif line.startswith("end_header"):
                break
        if fmt == "ascii":
            V = np.loadtxt(f, max_rows=n_v, dtype=np.float32)[:, :3]
        else:
            dt_map = {"float": "f4", "float32": "f4", "double": "f8",
                      "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}
            endian = "<" if "little" in fmt else ">"
            rec = np.dtype([(f"p{i}", endian + dt_map.get(t, "f4"))
                            for i, t in enumerate(vprops)])
            raw = np.frombuffer(f.read(rec.itemsize * n_v), dtype=rec, count=n_v)
            V = np.stack([raw["p0"], raw["p1"], raw["p2"]], 1).astype(np.float32)
    if len(V) > max_v:
        idx = np.random.default_rng(0).permutation(len(V))[:max_v]
        V = V[idx]
    return V


def max_chord_2d(pts: np.ndarray) -> tuple[float, int, int]:
    """2D 점군 최대 chord (convex hull 정점 쌍 전수)."""
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        return 0.0, 0, 0
    hull = ConvexHull(pts)
    hv = hull.vertices
    P = pts[hv]
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return float(np.sqrt(d2[i, j])), int(hv[i]), int(hv[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--imgs", type=int, default=5)
    ap.add_argument("--visib", type=float, default=0.95)
    args = ap.parse_args()

    from PIL import Image
    models_dir = TLESS / "models_eval"
    if not models_dir.exists():
        models_dir = TLESS / "models_cad"
    scene_dirs = sorted((TLESS / "test_primesense").iterdir())[: args.scenes]
    print(f"=== N2 T-LESS 측정 상한 ===  scenes={len(scene_dirs)}, models={models_dir.name}")

    model_cache: dict[int, np.ndarray] = {}
    rows, rels = [], []
    for sd in scene_dirs:
        cam = json.loads((sd / "scene_camera.json").read_text())
        gt = json.loads((sd / "scene_gt.json").read_text())
        gti = json.loads((sd / "scene_gt_info.json").read_text())
        im_ids = sorted(gt.keys(), key=int)[: args.imgs]
        for im in im_ids:
            K = np.array(cam[im]["cam_K"]).reshape(3, 3)
            fx = K[0, 0]
            for gi, (obj, info) in enumerate(zip(gt[im], gti[im])):
                if info.get("visib_fract", 0) < args.visib:
                    continue
                oid = obj["obj_id"]
                if oid not in model_cache:
                    model_cache[oid] = load_ply_vertices(models_dir / f"obj_{oid:06d}.ply")
                V = model_cache[oid]
                R = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj["cam_t_m2c"]).reshape(3)
                Xc = V @ R.T + t                          # [N,3] mm (카메라계)
                P = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2],
                              K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
                px_proj, i, j = max_chord_2d(P)
                gt_mm = float(np.linalg.norm(Xc[i] - Xc[j]))
                # GT visible mask 최대 chord
                mp = sd / "mask_visib" / f"{int(im):06d}_{gi:06d}.png"
                if not mp.exists():
                    continue
                m = np.array(Image.open(mp)) > 0
                ys, xs = np.nonzero(m)
                if len(xs) < 30:
                    continue
                px_mask, _, _ = max_chord_2d(np.stack([xs, ys], 1).astype(np.float64))
                pred_mm = px_mask * float(t[2]) / fx       # plane-scale (Z=t_z)
                rel = abs(pred_mm - gt_mm) / max(gt_mm, 1e-6)
                rels.append(rel)
                rows.append({"scene": sd.name, "im": im, "obj": oid,
                             "gt_mm": round(gt_mm, 2), "pred_mm": round(pred_mm, 2),
                             "rel": round(rel, 4),
                             "px_mask_vs_proj": round(px_mask / max(px_proj, 1e-6), 4)})

    rel = np.array(rels)
    by_obj = {}
    for r in rows:
        by_obj.setdefault(r["obj"], []).append(r["rel"])
    summary = {"n": len(rows),
               "rel_err_mean": round(float(rel.mean()), 4),
               "rel_err_median": round(float(np.median(rel)), 4),
               "pass@5pct": round(float((rel <= 0.05).mean()), 3),
               "pass@10pct": round(float((rel <= 0.10).mean()), 3),
               "worst": round(float(rel.max()), 4),
               "per_obj_median": {str(k): round(float(np.median(v)), 4)
                                  for k, v in sorted(by_obj.items())},
               "note": "GT mask + pose-depth plane scale — 방법론적 상한 (분할 난이도 제외)"}
    print(f"\nn={summary['n']}  rel_err 평균 {summary['rel_err_mean']*100:.2f}% / "
          f"중앙값 {summary['rel_err_median']*100:.2f}%  ±5% {summary['pass@5pct']*100:.0f}% "
          f"±10% {summary['pass@10pct']*100:.0f}%  worst {summary['worst']*100:.1f}%")
    print("객체별 중앙값(상위 6):",
          dict(list(summary["per_obj_median"].items())[:6]))
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "tless_upper_bound.json").write_text(json.dumps(
        {"summary": summary, "rows": rows[:200]}, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/tless_upper_bound.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
