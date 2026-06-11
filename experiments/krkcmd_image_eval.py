"""E-mm-3b — krkCMd 이미지 레벨: prompt→mask→μm 체인의 물리 GT 검증.

E-mm-3(profile-level)의 승격판. 스캐너 이미지에서:
  rung1 (체인 검증): 테이블 밝기 벡터(x1..x501)를 이미지 내 상관 매칭으로 위치 복원
    → 같은 위치에서 우리가 추출한 프로파일에 minrun5(+cal) 적용 → MANwidth 비교.
    (위치 복원이 정확하면 profile-level 수치와 일치해야 함 — sanity)
  rung2 (promptable): SAM3 "crack" (다운스케일 추론→업스케일 마스크) → 마스크 EDT 폭을
    각 gridline 열에서 추출 → ×3.96875μm/px → MANwidth 비교.
    **이것이 "promptable image → μm"의 첫 물리 GT 수치.**

위치 복원: 열 후보 = xS + 100·g (spacingL=100, ImageJ 스크립트). xS는 이미지별 미지
→ 소수 프로파일로 전 열 스캔해 mod-100 오프셋 추정, 이후 열별 y는 1D 상관 argmax.

실행: python experiments/krkcmd_image_eval.py --series CMd_0.23_2mths --image 1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KRK = Path("datasets/krkcmd")
PX_TO_UM = 1000.0 * 25.4 / 6400.0
N_P = 501


def load_rows(series: str, image: str):
    rows = []
    with open(KRK / "krkCMd_table.csv") as f:
        for row in csv.DictReader(f):
            if row["Series"] == series and row["Image"] == image:
                gy, gx = (int(v) for v in row["Gridline"].split("-"))
                rows.append({
                    "gx": gx, "gy": gy, "stage": int(row["Stage"]),
                    "x": np.array([float(row[f"x{i}"]) for i in range(1, N_P + 1)], np.float32),
                    "man": float(row["MANwidth"]), "dlm": float(row["DLMwidth"])})
    return rows


def load_stack(series: str, image: str):
    import tifffile
    p = KRK / series / f"{series}_Image{image}.tif"
    with tifffile.TiffFile(p) as tf:
        pages = [pg.asarray() for pg in tf.pages]
    return pages  # [stage] = (H, W) or (H, W, C)


def to_gray(a):
    return a.mean(-1).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)


def match_column(col: np.ndarray, vec: np.ndarray):
    """열 1D 신호에서 501-벡터 최적 위치 (정규화 상관)."""
    n = len(vec)
    if len(col) < n:
        return -1, -1.0
    v = (vec - vec.mean()) / (vec.std() + 1e-6)
    # 슬라이딩 윈도우 정규화 상관 (FFT 없이도 H~수천이면 충분)
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(col, n)                      # [H-n+1, n]
    Wm = W - W.mean(1, keepdims=True)
    denom = W.std(1) + 1e-6
    corr = (Wm @ v) / (denom * n)
    y0 = int(np.argmax(corr))
    return y0, float(corr[y0])


def refine_y(col: np.ndarray, vec: np.ndarray, y_hint: int, win: int = 25):
    """Gridline y 좌표 주변 ±win에서 최적 상관 위치 미세보정."""
    lo = max(0, y_hint - win)
    hi = min(len(col) - N_P, y_hint + win)
    if hi <= lo:
        return y_hint, -1.0
    y0, score = match_column(col[lo: hi + N_P], vec)
    return lo + y0, score


def mask_width_at_col(mask: np.ndarray, c: int, halfwin: int = 2) -> float:
    """열 c 주변 마스크 폭(px): 열 구간 내 최대 연속 run."""
    seg = mask[:, max(0, c - halfwin): c + halfwin + 1].any(1)
    if not seg.any():
        return 0.0
    runs, cur = [], 0
    for v in seg:
        cur = cur + 1 if v else 0
        runs.append(cur)
    return float(max(runs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CMd_0.23_2mths")
    ap.add_argument("--image", default="1")
    ap.add_argument("--stages", type=int, default=3, help="평가할 stage 수 (페이지)")
    ap.add_argument("--downscale", type=int, default=4, help="SAM3 추론 다운스케일 배율")
    ap.add_argument("--skip-sam3", action="store_true")
    args = ap.parse_args()

    rows_all = load_rows(args.series, args.image)
    pages = load_stack(args.series, args.image)
    print(f"=== E-mm-3b {args.series}/Image{args.image}: pages={len(pages)}, "
          f"table rows={len(rows_all)} ===")

    from experiments.krkcmd_profile_eval import width_min_run  # 동일 규칙 재사용
    res = {"rung1": [], "rung2": []}
    for stage in sorted({r["stage"] for r in rows_all})[: args.stages]:
        rows = [r for r in rows_all if r["stage"] == stage]
        if stage - 1 >= len(pages) or not rows:
            continue
        gray = to_gray(pages[stage - 1])
        H, W = gray.shape
        print(f"[stage {stage}] {H}x{W}, profiles={len(rows)} (Gridline=yyyy-xxxx 직접 좌표)")

        # rung1: 위치 복원 → 우리 추출 프로파일 → minrun5
        r1_err, matched = [], 0
        pos = {}
        for r in rows:
            c = r["gx"]
            if c >= W:
                continue
            y0, score = refine_y(gray[:, c], r["x"], r["gy"])
            if score < 0.95:    # 위치 신뢰 게이트 (상관 검증)
                continue
            matched += 1
            pos[(r["gx"], r["gy"])] = (c, y0)
            prof = gray[y0: y0 + N_P, c]
            w_um = width_min_run(prof)
            r1_err.append(abs(w_um - r["man"]))
        if r1_err:
            res["rung1"].append({"stage": stage, "n": len(r1_err),
                                 "match_rate": round(matched / len(rows), 3),
                                 "MAE_um": round(float(np.mean(r1_err)), 1)})
            print(f"  rung1: 복원 {matched}/{len(rows)} · 무보정 MAE {np.mean(r1_err):.1f}μm")

        # rung2: SAM3 "crack" 다운스케일 추론 → 마스크 폭
        if args.skip_sam3 or not pos:
            continue
        from gaugeanything.segmenters import segment_sam3
        ds = args.downscale
        g8 = np.clip(gray / max(gray.max(), 1) * 255, 0, 255).astype(np.uint8)
        small = g8[::ds, ::ds]
        img3 = np.stack([small] * 3, -1)
        insts = segment_sam3(img3, "crack", threshold=0.3)
        if not insts:
            print("  rung2: SAM3 crack 미검출")
            res["rung2"].append({"stage": stage, "n": 0, "note": "no detection"})
            continue
        m_small = np.zeros(small.shape, bool)
        for inst in insts:
            m_small |= inst.mask
        # 업스케일 (nearest)
        m_full = np.repeat(np.repeat(m_small, ds, 0), ds, 1)[:H, :W]
        r2_err, r2_n = [], 0
        for (gx, gy), (c, y0) in pos.items():
            man = next(r["man"] for r in rows if r["gx"] == gx and r["gy"] == gy)
            w_px = mask_width_at_col(m_full[y0: y0 + N_P, :], c)
            w_um = w_px * PX_TO_UM
            r2_n += 1
            r2_err.append(abs(w_um - man))
        if r2_err:
            res["rung2"].append({"stage": stage, "n": r2_n,
                                 "MAE_um": round(float(np.mean(r2_err)), 1),
                                 "median_um": round(float(np.median(r2_err)), 1)})
            print(f"  rung2 (SAM3 crack→mask→μm): n={r2_n} MAE {np.mean(r2_err):.1f}μm "
                  f"중앙 {np.median(r2_err):.1f}μm")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "krkcmd_image_eval.json").write_text(json.dumps(
        {"series": args.series, "image": args.image, "results": res,
         "anchors": {"profile_level_cal_MAE": 27.8, "DLM": 11.1}},
        indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/krkcmd_image_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
