"""Count v1 — ROI-1555 rebar density/centroid head.

근거 사슬: zero-shot SAM3는 dense touching bar에서 구조적으로 undercount한다
(E-cnt-1 best prompt MAE 13.1, E-cnt-2 SAHI MAE 8.9). 카운팅은 더 이상 prompt
문제가 아니라 표현 문제다 — 밀집 인스턴스는 detection보다 density 회귀가 맞다.

이 스크립트는 작은 owned density head를 학습한다:
  image -> tiny FCN -> 1-channel density map -> count = sum(map)

프로토콜
--------
- 데이터: ROI-1555 labelme (1,260 labeled, polygon shapes, GT count = #shapes).
- GT density: 각 폴리곤 centroid에 정규화 Gaussian(합=1) → 맵 합 = count.
- split: image id 정렬 후 contiguous block (train 70 / val 15 / test 15) —
  연속 프레임 near-duplicate leakage 차단 (random split 금지).
- 선택: val MAE 최소 epoch. test는 선택에 미사용.
- 지표: MAE, RMSE, acc@10%, dense-bin(GT>=40) MAE + undercount bias.
  앵커: SAHI-SAM3 zero-shot MAE 8.9 (E-cnt-2). 목표: held-out MAE < 5,
  dense-bin undercount 50%+ 감소.

Spark 실행:
    .venv/bin/python -u experiments/rebar_density_head.py --epochs 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("datasets/rebar_roi1555")
OUT = Path("experiments/results/rebar_density_head.json")
CKPT = Path("checkpoints/rebar_density_head.pt")

# 기본값 (v1). v1.1은 CLI로 고해상도/sharp sigma/count-weighted loss 오버라이드.
IN_W, IN_H = 1024, 640      # 고정 입력 (원본 1333x800, aspect 보존 근사)
STRIDE = 8                  # density map = 128 x 80
SIGMA = 2.0                 # downscaled 공간에서의 Gaussian sigma
DENSE_BIN = 40              # dense-bin 임계 (GT count)


def list_labeled() -> list[Path]:
    return sorted((ROOT / "1260" / "img_label").glob("*.json"))


def polygon_centroids(shapes: list[dict]) -> np.ndarray:
    pts = []
    for s in shapes:
        p = np.asarray(s["points"], dtype=np.float32)
        if len(p) >= 1:
            pts.append(p.mean(axis=0))   # 얇은 막대 → 점 평균 centroid로 충분
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)


def make_density(centroids: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """centroid(원본 px) → 고정 입력 리사이즈 → /STRIDE density map. 합 = count."""
    dw, dh = IN_W // STRIDE, IN_H // STRIDE
    dm = np.zeros((dh, dw), np.float32)
    if len(centroids) == 0:
        return dm
    sx, sy = IN_W / img_w, IN_H / img_h
    xs = np.clip(centroids[:, 0] * sx / STRIDE, 0, dw - 1)
    ys = np.clip(centroids[:, 1] * sy / STRIDE, 0, dh - 1)
    # 정규화 Gaussian splat: 각 점이 합 1을 기여하도록.
    yy, xx = np.mgrid[0:dh, 0:dw]
    for x, y in zip(xs, ys):
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2))
        s = g.sum()
        if s > 0:
            dm += g / s
    return dm


def build_dataset():
    import cv2

    jsons = list_labeled()
    imgs, dens, counts = [], [], []
    for jp in jsons:
        d = json.loads(jp.read_text())
        ip = jp.with_suffix(".jpg")
        if not ip.exists():
            continue
        cen = polygon_centroids(d["shapes"])
        im = cv2.imread(str(ip))
        if im is None:
            continue
        im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        imgs.append(im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        dens.append(make_density(cen, d["imageWidth"], d["imageHeight"]))
        counts.append(len(d["shapes"]))
    return np.asarray(imgs), np.asarray(dens), np.asarray(counts, np.int32)


def block_split(n: int):
    """id-정렬된 contiguous block split (jsons는 이미 정렬됨)."""
    i_tr = int(n * 0.70)
    i_va = int(n * 0.85)
    return np.arange(0, i_tr), np.arange(i_tr, i_va), np.arange(i_va, n)


def stratified_split(counts: np.ndarray, seed: int = 0):
    """count-bin stratified split: train/val/test의 밀도 분포를 맞춘다.

    block split이 우연히 test 블록을 고밀도로 몰아 분포 이동을 만든 confound를 제거.
    ROI-1555 라벨 이미지는 scene 메타데이터가 없어 scene-holdout이 불가능하므로,
    count 분위 bin 내에서 무작위 배분하는 것이 가장 방어 가능한 분할이다.
    (주의: 원본이 연속 프레임이면 near-duplicate 누수 가능 — 한계로 보고.)
    """
    rng = np.random.default_rng(seed)
    bins = np.digitize(counts, np.quantile(counts, [0.2, 0.4, 0.6, 0.8]))
    tr, va, te = [], [], []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        i_tr, i_va = int(len(idx) * 0.70), int(len(idx) * 0.85)
        tr.extend(idx[:i_tr]); va.extend(idx[i_tr:i_va]); te.extend(idx[i_va:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def build_net():
    import torch.nn as nn

    def cbr(i, o, d=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(True))

    # 출력 ReLU 없음: per-pixel density가 ~1e-3로 작아 ReLU 출력이 죽으면(dead)
    # 모델이 전부 0을 예측하는 zero-collapse에 빠진다. raw 출력 + count loss로 학습.
    return nn.Sequential(
        cbr(3, 32), nn.MaxPool2d(2),       # /2
        cbr(32, 64), nn.MaxPool2d(2),      # /4
        cbr(64, 64), nn.MaxPool2d(2),      # /8
        cbr(64, 64, 2), cbr(64, 32, 2),    # dilated context
        nn.Conv2d(32, 1, 1),
    )


def evaluate(pred_counts: np.ndarray, gt: np.ndarray) -> dict:
    err = pred_counts - gt
    ae = np.abs(err)
    dense = gt >= DENSE_BIN
    out = {
        "n": int(len(gt)),
        "MAE": round(float(ae.mean()), 3),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
        "bias": round(float(err.mean()), 3),
        "acc@10pct": round(float((ae <= 0.1 * np.maximum(gt, 1)).mean()), 3),
        "gt_mean": round(float(gt.mean()), 2),
        "pred_mean": round(float(pred_counts.mean()), 2),
    }
    if dense.any():
        out["dense_bin"] = {
            "n": int(dense.sum()),
            "MAE": round(float(ae[dense].mean()), 3),
            "bias": round(float(err[dense].mean()), 3),
            "gt_mean": round(float(gt[dense].mean()), 2),
        }
    return out


def main() -> int:
    global IN_W, IN_H, SIGMA
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["stratified", "block"], default="stratified")
    ap.add_argument("--in-w", type=int, default=1024)
    ap.add_argument("--in-h", type=int, default=640)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--count-weight", type=float, default=0.0,
                    help="count loss를 sqrt(gt)^w로 가중 (0=균일). dense 이미지 비중 상향.")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--save", type=Path, default=CKPT)
    args = ap.parse_args()

    IN_W, IN_H, SIGMA = args.in_w, args.in_h, args.sigma

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== Count v1 — rebar density head ===", flush=True)
    print("building dataset...", flush=True)
    X, D, C = build_dataset()
    print(f"images {len(X)} · gt count mean {C.mean():.1f} max {C.max()}", flush=True)
    tr, va, te = stratified_split(C, args.seed) if args.split == "stratified" else block_split(len(X))
    print(f"split ({args.split}): train {len(tr)} / val {len(va)} / test {len(te)} "
          f"| gt_mean tr {C[tr].mean():.1f} va {C[va].mean():.1f} te {C[te].mean():.1f}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net().to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"params {nparam/1e6:.2f}M · device {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xt = torch.from_numpy(X).to(dev)
    Dt = torch.from_numpy(D).unsqueeze(1).to(dev)
    Ct = torch.from_numpy(C.astype(np.float32)).to(dev)

    def count_pred(idx) -> np.ndarray:
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), args.batch):
                b = idx[i:i + args.batch]
                pm = net(Xt[b])
                out.append(pm.sum(dim=(1, 2, 3)).clamp(min=0).cpu().numpy())
        return np.concatenate(out)

    best_val, best_state, best_ep = 1e9, None, -1
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            pm = net(Xt[b])
            # count L1을 주 신호로 (magnitude ~21), density MSE는 공간 정규화기.
            # per-pixel density가 작아 MSE만으로는 zero-collapse → 큰 GAIN으로 증폭.
            dl = torch.nn.functional.mse_loss(pm * 1000, Dt[b] * 1000)
            ce = torch.abs(pm.sum(dim=(1, 2, 3)) - Ct[b])
            if args.count_weight > 0:
                # dense 이미지가 bulk에 묻히지 않도록 sqrt(gt) 가중 (count imbalance 대응).
                w = (Ct[b].clamp(min=1.0).sqrt()) ** args.count_weight
                cl = (ce * w).sum() / w.sum()
            else:
                cl = ce.mean()
            loss = cl + 0.1 * dl
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        val_mae = float(np.abs(count_pred(va) - C[va]).mean())
        if val_mae < best_val:
            best_val, best_ep = val_mae, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep{ep+1:3d} loss {tot/len(tr):.3f} val_MAE {val_mae:.2f} (best {best_val:.2f}@{best_ep+1})", flush=True)

    net.load_state_dict(best_state)
    test_pred = count_pred(te)
    val_pred = count_pred(va)
    test_eval = evaluate(test_pred, C[te])
    val_eval = evaluate(val_pred, C[va])

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "input": f"{IN_W}x{IN_H} RGB", "stride": STRIDE,
                "sigma": SIGMA, "selected_epoch": best_ep + 1}, args.save)

    result = {
        "protocol": {
            "task": "rebar count via density-map regression head over ROI-1555",
            "data": "ROI-1555 labelme (1260 labeled), GT count = #polygon shapes",
            "split": args.split,
            "split_note": ("stratified = count-bin stratified (밀도 분포 정합, scene 메타 없음); "
                           "block = id-정렬 contiguous (보수적이나 test 블록이 고밀도로 쏠림). "
                           "둘 다 연속 프레임이면 near-duplicate 누수 가능 — 한계."),
            "gt_mean_per_split": {"train": round(float(C[tr].mean()), 2),
                                   "val": round(float(C[va].mean()), 2),
                                   "test": round(float(C[te].mean()), 2)},
            "lr_schedule": "cosine annealing",
            "input_res": f"{IN_W}x{IN_H}", "sigma": SIGMA, "count_weight": args.count_weight,
            "selection": "min val MAE epoch; test labels unused for selection",
            "density": f"normalized Gaussian (sum=1) per centroid, stride {STRIDE}, sigma {SIGMA}",
            "params_M": round(nparam / 1e6, 3),
            "selected_epoch": best_ep + 1,
        },
        "anchors": {
            "sahi_sam3_zeroshot_MAE": 8.9,
            "sahi_sam3_note": "E-cnt-2, n=40 random sample; dense touching bars undercount (GT 81->40)",
            "target_MAE": 5.0,
        },
        "val": val_eval,
        "test": test_eval,
        "test_rows": [
            {"gt": int(g), "pred": round(float(p), 1)} for g, p in zip(C[te], test_pred)
        ][:40],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("test:", json.dumps(test_eval, ensure_ascii=False), flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"saved {CKPT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
