"""E-loop-0 — 결정적 active reacquisition 루프 (AGENTIC_LOOP_DESIGN 1단계).

1차 패스(타일 SAM3 스켈레톤)의 실패 구간을 GT 없이 식별(연속성 게이트)하고,
인접 신뢰 열의 행 보간을 prior로 풀해상도 줌 크롭에서 재분할 → 재스냅 → 게이트.
LLM 없는 루프 — VLM 에이전트(E-loop-2)의 기여를 분리하기 위한 베이스라인이자,
그 자체로 커버리지 회복 수단.

평가(oracle은 평가에만 사용): 커버리지(≤80px) 1차 vs 루프 후, gated CNN 폭 MAE.
성공 기준: 커버리지 46~66% → 80%+, gated MAE ≤50μm.

실행: python experiments/krkcmd_loop_recovery.py --stages 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.krkcmd_image_eval import N_P, load_rows, load_stack, to_gray, refine_y  # noqa: E402
from experiments.krkcmd_signal_width import (sam3_skeleton_rows, train_profile_cnn)  # noqa: E402

JUMP = 120          # 연속성 게이트: 경로 대비 허용 점프(px)
SNAP = 80           # snap-to-valley 반경


def smooth_path(cols: np.ndarray, ys: np.ndarray, q_cols: np.ndarray) -> np.ndarray:
    """수락된 (col,y)로부터 질의 열의 경로 prior (이동 중앙값 + 선형 보간)."""
    order = np.argsort(cols)
    c_s, y_s = cols[order], ys[order].astype(float)
    if len(c_s) >= 7:           # 이동 중앙값으로 잔존 아웃라이어 완화
        k = 5
        y_med = np.array([np.median(y_s[max(0, i - k): i + k + 1]) for i in range(len(y_s))])
    else:
        y_med = y_s
    return np.interp(q_cols, c_s, y_med)


def snap_valley(gray: np.ndarray, c: int, y_hint: float) -> int:
    lo = max(0, int(y_hint) - SNAP)
    hi = min(gray.shape[0], int(y_hint) + SNAP + 1)
    return lo + int(np.argmin(gray[lo:hi, c]))


def zoom_reacquire(gray: np.ndarray, c: int, y_prior: float, half: int,
                   prompts=("crack",)) -> int | None:
    """prior 주변 풀해상도 크롭에서 SAM3 재분할 → 중심열 부근 스켈레톤 행."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from gaugeanything.segmenters import segment_sam3
    H, W = gray.shape
    y0, y1 = max(0, int(y_prior) - half), min(H, int(y_prior) + half)
    x0, x1 = max(0, c - half), min(W, c + half)
    crop = gray[y0:y1, x0:x1]
    if crop.size < 1000:
        return None
    g8 = np.clip(crop / max(crop.max(), 1) * 255, 0, 255).astype(np.uint8)
    m = np.zeros(crop.shape, bool)
    for p in prompts:
        for inst in segment_sam3(np.stack([g8] * 3, -1), p, threshold=0.3):
            m |= inst.mask
    if not m.any():
        return None
    lab, n = ndimage.label(m)
    means = ndimage.mean(g8, lab, np.arange(1, n + 1))
    sk = skeletonize(lab == (1 + int(np.argmin(means))))   # 최암 성분
    cc = c - x0
    band = sk[:, max(0, cc - 4): cc + 5]
    ys = np.nonzero(band.any(1))[0]
    if not len(ys):
        return None
    return y0 + int(np.median(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CMd_0.23_2mths")
    ap.add_argument("--image", default="1")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--col-tol", type=int, default=15)
    args = ap.parse_args()

    print("=== E-loop-0 결정적 active reacquisition ===")
    predict_cnn, _ = train_profile_cnn(epochs=12)
    rows_all = load_rows(args.series, args.image)
    pages = load_stack(args.series, args.image)

    res = {"first_pass": {"near": 0, "n": 0, "errs": []},
           "after_loop": {"near": 0, "n": 0, "errs": []},
           "loop_log": {"recovered_zoom1": 0, "recovered_zoom2": 0,
                        "fallback_snap": 0, "rejected": 0}}
    for stage in sorted({r["stage"] for r in rows_all})[: args.stages]:
        rows = [r for r in rows_all if r["stage"] == stage]
        gray = to_gray(pages[stage - 1])
        H, W = gray.shape
        skel = sam3_skeleton_rows(gray)
        skel_cols = np.array(sorted(skel)) if skel else np.array([])

        # oracle (평가 전용)
        oracle = {}
        for r in rows:
            c = r["gx"]
            if c >= W:
                continue
            y0, sc = refine_y(gray[:, c], r["x"], r["gy"] - 250, win=60)
            if sc >= 0.95:
                oracle[c] = (y0 + 250, r["man"])

        # 1차 후보
        cand = {}
        for c in oracle:
            if not len(skel_cols):
                continue
            j = int(np.argmin(np.abs(skel_cols - c)))
            if abs(int(skel_cols[j]) - c) <= args.col_tol:
                cand[c] = snap_valley(gray, c, skel[int(skel_cols[j])])

        # 연속성 게이트로 신뢰/실패 분리 (GT 불사용)
        cols = np.array(sorted(cand))
        ys = np.array([cand[c] for c in cols], float)
        trusted = {}
        if len(cols) >= 5:
            path = smooth_path(cols, ys, cols.astype(float))
            for c, y, p in zip(cols, ys, path):
                if abs(y - p) <= JUMP:
                    trusted[int(c)] = int(y)
        failed = [c for c in oracle if c not in trusted]

        # 1차 평가
        for c, (yo, man) in oracle.items():
            res["first_pass"]["n"] += 1
            y = cand.get(c)
            if y is not None and abs(y - yo) <= 80:
                res["first_pass"]["near"] += 1
                a = max(0, snap_valley(gray, c, y) - N_P // 2)
                p = gray[a: a + N_P, c]
                if len(p) == N_P:
                    res["first_pass"]["errs"].append(
                        abs(float(predict_cnn([p])[0]) - man))

        # ----- 복구 루프 -----
        final = dict(trusted)
        if trusted and failed:
            t_cols = np.array(sorted(trusted))
            t_ys = np.array([trusted[c] for c in t_cols], float)
            priors = smooth_path(t_cols, t_ys, np.array(sorted(failed), float))
            for c, pr in zip(sorted(failed), priors):
                got = None
                y1 = zoom_reacquire(gray, c, pr, half=256)
                if y1 is not None and abs(y1 - pr) <= 150:
                    got = y1; res["loop_log"]["recovered_zoom1"] += 1
                else:
                    y2 = zoom_reacquire(gray, c, pr, half=384,
                                        prompts=("crack", "thin dark line"))
                    if y2 is not None and abs(y2 - pr) <= 150:
                        got = y2; res["loop_log"]["recovered_zoom2"] += 1
                    else:
                        ysnap = snap_valley(gray, c, pr)
                        # 명암 게이트: valley가 주변 대비 충분히 어두운가
                        seg = gray[max(0, ysnap - 250): ysnap + 251, c]
                        if len(seg) > 100 and (np.median(seg) - seg.min()) > 3 * seg.std() * 0.5:
                            got = ysnap; res["loop_log"]["fallback_snap"] += 1
                if got is None:
                    res["loop_log"]["rejected"] += 1
                else:
                    final[c] = snap_valley(gray, c, got)

        # 루프 후 평가
        for c, (yo, man) in oracle.items():
            res["after_loop"]["n"] += 1
            y = final.get(c)
            if y is not None and abs(y - yo) <= 80:
                res["after_loop"]["near"] += 1
                a = max(0, y - N_P // 2)
                p = gray[a: a + N_P, c]
                if len(p) == N_P:
                    res["after_loop"]["errs"].append(
                        abs(float(predict_cnn([p])[0]) - man))
        print(f"[stage {stage}] oracle n={len(oracle)} 1차후보 {len(cand)} "
              f"신뢰 {len(trusted)} 실패 {len(failed)} → 최종 {len(final)}")

    for k in ("first_pass", "after_loop"):
        d = res[k]
        cov = d["near"] / max(d["n"], 1)
        mae = float(np.mean(d["errs"])) if d["errs"] else float("nan")
        med = float(np.median(d["errs"])) if d["errs"] else float("nan")
        d.update({"coverage": round(cov, 3), "MAE_um": round(mae, 1),
                  "median_um": round(med, 1)})
        print(f"{k:>11}: 커버리지 {cov*100:.0f}%  gated MAE {mae:.1f}μm  중앙 {med:.1f}μm "
              f"(n={len(d['errs'])})")
    print("루프 로그:", res["loop_log"])
    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    drop = {k: {kk: vv for kk, vv in v.items() if kk != "errs"} if isinstance(v, dict) else v
            for k, v in res.items()}
    (out / "krkcmd_loop_recovery.json").write_text(json.dumps(drop, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/krkcmd_loop_recovery.json")
    print("성공 기준: 커버리지 80%+ & gated MAE ≤50μm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
