"""E-loop-1 — 다중 후보 경로 + 크랙다움 점수 (결정적 의미 선택).

E-loop-0 진단: 오인 구조가 매끈한 경로를 이뤄 연속성 게이트 무력 → 실패의 본질은
"어느 어두운 선이 크랙인가"(의미). VLM 전에 결정적 의미 점수로 재도전:

1. 타일 SAM3 마스크의 **모든 유의 성분**을 후보 경로로 유지 (폭 15%+).
2. 각 경로에서 M개 등간격 프로파일 추출 → **크랙다움 점수** (GT-free 물리 사전):
   - valley 대비: (배경중앙값 − 최솟값) / 국소 std  (깊고 선명)
   - half-depth 폭 ∈ [8, 250]px  (크랙 폭 물리 범위 — 가장자리 그림자는 넓음)
   - 배경 대칭성: 좌/우 25% 구간 중앙값 차 / 대비  (가장자리는 비대칭)
3. 열별로 최고 점수 경로의 행 채택 (경로 단위가 아닌 열 단위 선택 — 크랙 전환 허용).
4. snap-to-valley → 기존 게이트 → 평가(oracle, 평가 전용).

실행: python experiments/krkcmd_loop_pathselect.py --stages 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.krkcmd_image_eval import N_P, load_rows, load_stack, to_gray, refine_y  # noqa: E402
from experiments.krkcmd_signal_width import train_profile_cnn  # noqa: E402
from experiments.krkcmd_profile_eval import width_half_depth  # noqa: E402

SNAP = 80


def crackness(profile: np.ndarray) -> float:
    """GT-free 크랙다움 점수 (물리 사전). 높을수록 크랙."""
    p = profile.astype(np.float64)
    n = len(p)
    nb = max(10, n // 4)
    bg_l, bg_r = np.median(p[:nb]), np.median(p[-nb:])
    bg = 0.5 * (bg_l + bg_r)
    depth = bg - p.min()
    std = max(p.std(), 1e-6)
    contrast = depth / std                       # 깊고 선명할수록 ↑
    w_hd = width_half_depth(p) / (25.4 / 6400 * 1000)   # px 단위 half-depth 폭
    width_ok = 1.0 if 8 <= w_hd <= 250 else 0.2
    asym = abs(bg_l - bg_r) / max(depth, 1e-6)   # 가장자리 그림자=비대칭
    sym_ok = 1.0 / (1.0 + 2.0 * asym)
    return float(contrast * width_ok * sym_ok)


def multi_component_paths(gray: np.ndarray, ds: int = 2, tile: int = 1024,
                          overlap: float = 0.2, min_span: float = 0.15):
    """타일 SAM3 → 유의 성분 전부의 (열→행) 경로 목록."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from gaugeanything.segmenters import segment_sam3
    g8 = np.clip(gray / max(gray.max(), 1) * 255, 0, 255).astype(np.uint8)
    small = g8[::ds, ::ds]
    h, w = small.shape
    m = np.zeros(small.shape, bool)
    stride = max(1, int(tile * (1 - overlap)))
    xs = sorted({min(x, max(w - tile, 0)) for x in range(0, max(w - tile, 0) + stride, stride)})
    ys = sorted({min(y, max(h - tile, 0)) for y in range(0, max(h - tile, 0) + stride, stride)})
    for y0 in ys:
        for x0 in xs:
            crop = small[y0:y0 + tile, x0:x0 + tile]
            for inst in segment_sam3(np.stack([crop] * 3, -1), "crack", threshold=0.3):
                m[y0:y0 + tile, x0:x0 + tile] |= inst.mask
    lab, n = ndimage.label(m)
    paths = []
    for k in range(1, n + 1):
        comp = lab == k
        xs_k = np.nonzero(comp.any(0))[0]
        if not len(xs_k) or (xs_k[-1] - xs_k[0]) < min_span * w:
            continue
        sk = skeletonize(comp)
        yy, xx = np.nonzero(sk)
        path = {}
        for y, x in zip(yy, xx):
            path.setdefault(int(x) * ds, []).append(int(y) * ds)
        paths.append({c: int(np.median(v)) for c, v in path.items()})
    return paths


def valley_candidates(col: np.ndarray, k: int = 5, min_gap: int = 100) -> list[int]:
    """열의 국소 최암점 상위 k개 (SAM3-독립 후보 — recall 보강)."""
    from scipy.ndimage import minimum_filter1d
    sm = minimum_filter1d(col, size=9)
    is_min = (col <= sm + 1e-6)
    idx = np.nonzero(is_min)[0]
    if not len(idx):
        return []
    order = idx[np.argsort(col[idx])]
    picked = []
    for y in order:
        if all(abs(int(y) - q) >= min_gap for q in picked):
            picked.append(int(y))
        if len(picked) >= k:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CMd_0.23_2mths")
    ap.add_argument("--image", default="1")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--col-tol", type=int, default=15)
    args = ap.parse_args()

    print("=== E-loop-1 다중 경로 + 크랙다움 점수 ===")
    predict_cnn, _ = train_profile_cnn(epochs=12)
    rows_all = load_rows(args.series, args.image)
    pages = load_stack(args.series, args.image)

    near = 0; n_tot = 0; errs = []
    n_paths_log = []
    for stage in sorted({r["stage"] for r in rows_all})[: args.stages]:
        rows = [r for r in rows_all if r["stage"] == stage]
        gray = to_gray(pages[stage - 1])
        H, W = gray.shape
        paths = multi_component_paths(gray, min_span=0.05)
        n_paths_log.append(len(paths))

        oracle = {}
        for r in rows:
            c = r["gx"]
            if c >= W:
                continue
            y0, sc = refine_y(gray[:, c], r["x"], r["gy"] - 250, win=60)
            if sc >= 0.95:
                oracle[c] = (y0 + 250, r["man"])

        # ---- 후보 수집 (열별) ----
        col_list = sorted(oracle)
        node_y, node_s = [], []          # 열별 후보 행/점수
        for c in col_list:
            cand_rows = []
            for path in paths:
                cols = np.array(sorted(path))
                if not len(cols):
                    continue
                j = int(np.argmin(np.abs(cols - c)))
                if abs(int(cols[j]) - c) <= args.col_tol:
                    cand_rows.append(path[int(cols[j])])
            cand_rows += valley_candidates(gray[:, c])
            ys_c, ss_c = [], []
            seen = set()
            for yh in cand_rows:
                lo = max(0, yh - SNAP); hi = min(H, yh + SNAP + 1)
                yc = lo + int(np.argmin(gray[lo:hi, c]))
                if yc in seen:
                    continue
                seen.add(yc)
                a = max(0, yc - N_P // 2)
                p = gray[a: a + N_P, c]
                if len(p) != N_P:
                    continue
                ys_c.append(yc); ss_c.append(crackness(p))
            node_y.append(ys_c); node_s.append(ss_c)

        # ---- Viterbi: 노드=log crackness, 전이=-λ|Δy|/Δc ----
        LAM = 8.0
        dp, bk = [], []
        for i, (ys_c, ss_c) in enumerate(zip(node_y, node_s)):
            sc = [np.log(max(v, 1e-6)) for v in ss_c]
            if i == 0 or not dp or not dp[-1]:
                dp.append(sc); bk.append([-1] * len(sc)); continue
            dc = max(col_list[i] - col_list[i - 1], 1) / 100.0
            row_dp, row_bk = [], []
            for yj, sj in zip(ys_c, sc):
                best_v, best_k = -1e18, -1
                for k, (yk, vk) in enumerate(zip(node_y[i - 1], dp[i - 1])):
                    v = vk - LAM * abs(yj - yk) / (dc * 1000.0)
                    if v > best_v:
                        best_v, best_k = v, k
                row_dp.append(best_v + sj); row_bk.append(best_k)
            dp.append(row_dp); bk.append(row_bk)
        # 역추적
        chosen = [None] * len(col_list)
        last = max(range(len(dp) - 1, -1, -1), key=lambda i: 0 if not dp[i] else 1)
        if dp and dp[-1]:
            k = int(np.argmax(dp[-1]))
            for i in range(len(col_list) - 1, -1, -1):
                if not dp[i]:
                    continue
                chosen[i] = node_y[i][k] if k < len(node_y[i]) else None
                k = bk[i][k] if k < len(bk[i]) and bk[i][k] >= 0 else (int(np.argmax(dp[i-1])) if i > 0 and dp[i-1] else 0)

        for (c, ych) in zip(col_list, chosen):
            yo, man = oracle[c]
            n_tot += 1
            if ych is None:
                continue
            if abs(ych - yo) <= 80:
                near += 1
                a = max(0, ych - N_P // 2)
                p = gray[a: a + N_P, c]
                errs.append(abs(float(predict_cnn([p])[0]) - man))
        print(f"[stage {stage}] paths={len(paths)} oracle n={len(oracle)}")

    cov = near / max(n_tot, 1)
    mae = float(np.mean(errs)) if errs else float("nan")
    med = float(np.median(errs)) if errs else float("nan")
    print(f"\nE-loop-1c(Viterbi): 커버리지 {cov*100:.0f}% (E-loop-0: 52%) · gated MAE {mae:.1f}μm "
          f"중앙 {med:.1f}μm (n={len(errs)})")
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "krkcmd_loop_pathselect.json").write_text(json.dumps(
        {"coverage": round(cov, 3), "MAE_um": round(mae, 1), "median_um": round(med, 1),
         "n": len(errs), "n_paths_per_stage": n_paths_log,
         "anchor": {"loop0_coverage": 0.52, "first_pass_MAE": 40.4}},
        indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/krkcmd_loop_pathselect.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
