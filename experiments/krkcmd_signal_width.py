"""E-mm-3c — 신호 기반 폭 추정: "mask for WHERE, signal for HOW WIDE" 결정 실험.

가설(WIDTH_BOTTLENECK_ANALYSIS): 병목은 마스크 품질이 아니라 '마스크 기하에서 폭을
읽는' 설계. 마스크는 중심선 위치만 제공하고, 폭은 full-res 원신호에서 직접 추정한다.

위치 2모드 (실패 원인 분리):
  oracle-pos : 검증된 테이블 좌표 (추정기 품질만 시험)
  sam3-pos   : SAM3 mask 스켈레톤이 주는 행 위치 (end-to-end promptable)

추정기 4종:
  A1 minrun5      — 기존 규칙 (rung1 35μm 앵커)
  A2 EW           — equivalent width: deficit 적분/깊이, PSF 1차 불변 (신규)
  A3 half-depth   — 반깊이 교차 (기존 규칙)
  A4 1D CNN       — krkCMd table train-split(그룹 분할)로 학습한 프로파일 회귀 (신규)
                    앵커: 저자 DLM 11.1μm / table-test에서 함께 보고

성공 기준: sam3-pos에서 ≤60μm (rung1 근접) → 제품 경로 개통.
실행: python experiments/krkcmd_signal_width.py --stages 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.krkcmd_image_eval import (PX_TO_UM, N_P, load_rows, load_stack,  # noqa: E402
                                           to_gray, refine_y)
from experiments.krkcmd_profile_eval import (read_rows, group_split,  # noqa: E402
                                             width_min_run, width_half_depth)


# ---------- A2: equivalent width (PSF 1차 불변) ----------
def width_equivalent(profile: np.ndarray, win: int = 150) -> float:
    c = int(np.argmin(profile))
    lo, hi = max(0, c - win), min(len(profile), c + win + 1)
    seg = profile[lo:hi].astype(np.float64)
    n_bg = max(10, len(seg) // 4)
    bg = float(np.median(np.concatenate([seg[:n_bg], seg[-n_bg:]])))
    depth = bg - float(seg.min())
    if depth <= 1e-6:
        return 0.0
    ew_px = float(np.clip(bg - seg, 0, None).sum() / depth)
    return ew_px * PX_TO_UM


# ---------- A4: 1D CNN (krkCMd table train-split 학습) ----------
def build_1d_net():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 96, 9, padding=4), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        nn.Flatten(), nn.Linear(96 * 8, 128), nn.ReLU(), nn.Linear(128, 1))


def norm_profile(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def train_profile_cnn(epochs: int = 12, seed: int = 0):
    import torch

    rows = read_rows()
    tr_idx, te_idx = group_split(rows)
    X = np.stack([norm_profile(r["x"]) for r in rows])
    y = np.array([r["MANwidth"] for r in rows], np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = build_1d_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    Xt = torch.from_numpy(X[tr_idx]).unsqueeze(1).to(dev)
    yt = torch.from_numpy(y[tr_idx]).to(dev)
    n = len(tr_idx)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            pred = net(Xt[idx]).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(pred, yt[idx], beta=20.0)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 4 == 0:
            print(f"  [1D CNN] ep{ep+1} train loss {tot/n:.2f}")
    net.eval()
    with torch.no_grad():
        pe = net(torch.from_numpy(X[te_idx]).unsqueeze(1).to(dev)).squeeze(1).cpu().numpy()
    mae = float(np.abs(pe - y[te_idx]).mean())
    print(f"  [1D CNN] table test MAE {mae:.1f}μm  (앵커: DLM 11.1 / minrun5+cal 25.9)")

    def predict(profiles: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            Z = torch.from_numpy(np.stack([norm_profile(p) for p in profiles])
                                 ).unsqueeze(1).to(dev)
            return net(Z).squeeze(1).cpu().numpy()
    return predict, mae


# ---------- SAM3 스켈레톤 위치 ----------
def sam3_skeleton_rows(gray: np.ndarray, ds: int = 2, tile: int = 1024, overlap: float = 0.2):
    """SAM3 'crack' 타일 추론(SAHI) → 최암·최장 성분 → 스켈레톤 → 열별 중심 행.

    진단(2026-06-12): 전역 추론은 내부 리사이즈로 크랙이 ~7px가 되어 다른 어두운
    구조를 잡음(중앙 1090px 오프). 타일에서는 크랙이 충분히 두꺼움. 성분 선택은
    promptable 원칙 유지(GT 불사용): 이미지 폭 30%+ 성분 중 평균 밝기 최암."""
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
            insts = segment_sam3(np.stack([crop] * 3, -1), "crack", threshold=0.3)
            for inst in insts:
                m[y0:y0 + tile, x0:x0 + tile] |= inst.mask
    if not m.any():
        return {}
    lab, n = ndimage.label(m)
    best, best_dark = None, 1e9
    for k in range(1, n + 1):
        comp = lab == k
        xs_k = np.nonzero(comp.any(0))[0]
        if len(xs_k) and (xs_k[-1] - xs_k[0]) >= 0.3 * w:
            dark = float(small[comp].mean())
            if dark < best_dark:
                best, best_dark = comp, dark
    if best is None:   # 폴백: 최암 성분
        means = ndimage.mean(small, lab, np.arange(1, n + 1))
        best = lab == (1 + int(np.argmin(means)))
    sk = skeletonize(best)
    ys, xs = np.nonzero(sk)
    col_rows: dict[int, list[int]] = {}
    for y, x in zip(ys, xs):
        col_rows.setdefault(int(x) * ds, []).append(int(y) * ds)
    return {c: int(np.median(v)) for c, v in col_rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="CMd_0.23_2mths")
    ap.add_argument("--image", default="1")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--col-tol", type=int, default=15, help="sam3 스켈레톤 열 허용 오차")
    args = ap.parse_args()

    print("=== E-mm-3c 신호 기반 폭 (mask=WHERE, signal=WIDTH) ===")
    predict_cnn, table_mae = train_profile_cnn()

    rows_all = load_rows(args.series, args.image)
    pages = load_stack(args.series, args.image)
    EST = {"A1_minrun5": width_min_run, "A2_EW": width_equivalent,
           "A3_halfdepth": width_half_depth}
    agg = {m: {k: [] for k in [*EST, "A4_cnn"]} for m in ("oracle", "sam3")}

    for stage in sorted({r["stage"] for r in rows_all})[: args.stages]:
        rows = [r for r in rows_all if r["stage"] == stage]
        if stage - 1 >= len(pages):
            continue
        gray = to_gray(pages[stage - 1])
        H, W = gray.shape
        skel = sam3_skeleton_rows(gray)
        skel_cols = np.array(sorted(skel)) if skel else np.array([])

        profs = {"oracle": [], "sam3": []}
        gts = {"oracle": [], "sam3": []}
        for r in rows:
            c = r["gx"]
            if c >= W:
                continue
            y0, score = refine_y(gray[:, c], r["x"], r["gy"] - 250, win=60)
            if score >= 0.95:
                profs["oracle"].append(gray[y0: y0 + N_P, c])
                gts["oracle"].append(r["man"])
            if len(skel_cols):
                j = int(np.argmin(np.abs(skel_cols - c)))
                if abs(int(skel_cols[j]) - c) <= args.col_tol:
                    yc = skel[int(skel_cols[j])]
                    # snap-to-valley: 스켈레톤 행 ±80px에서 최암점으로 재중심화
                    lo = max(0, yc - 80); hi = min(H, yc + 81)
                    yc = lo + int(np.argmin(gray[lo:hi, c]))
                    a = max(0, yc - N_P // 2)
                    p = gray[a: a + N_P, c]
                    if len(p) == N_P:
                        profs["sam3"].append(p)
                        gts["sam3"].append(r["man"])

        line = [f"[stage {stage}]"]
        for mode in ("oracle", "sam3"):
            if not profs[mode]:
                continue
            gt = np.array(gts[mode])
            for name, fn in EST.items():
                pred = np.array([fn(p) for p in profs[mode]])
                agg[mode][name].extend(np.abs(pred - gt).tolist())
            pred = predict_cnn(profs[mode])
            agg[mode]["A4_cnn"].extend(np.abs(pred - gt).tolist())
            line.append(f"{mode} n={len(gt)}")
        print(" ".join(line))

    out = {"table_cnn_MAE": round(table_mae, 1), "anchors":
           {"DLM": 11.1, "rung1_minrun_oracle": "35-43", "rung2_mask_best": "144-186"}}
    print(f"\n{'추정기':<14}{'oracle MAE':>12}{'sam3 MAE':>11}   (μm)")
    print("-" * 42)
    for name in ["A1_minrun5", "A2_EW", "A3_halfdepth", "A4_cnn"]:
        o = np.mean(agg["oracle"][name]) if agg["oracle"][name] else float("nan")
        s = np.mean(agg["sam3"][name]) if agg["sam3"][name] else float("nan")
        out[name] = {"oracle_MAE": round(float(o), 1), "sam3_MAE": round(float(s), 1),
                     "n_oracle": len(agg["oracle"][name]), "n_sam3": len(agg["sam3"][name])}
        print(f"{name:<14}{o:>12.1f}{s:>11.1f}")
    res = Path("experiments/results"); res.mkdir(parents=True, exist_ok=True)
    (res / "krkcmd_signal_width.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/krkcmd_signal_width.json")
    print("성공 기준: sam3 모드 ≤60μm → 'mask=WHERE, signal=WIDTH' 채택")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
