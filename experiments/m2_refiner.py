"""M2 — 측정 인식 refiner: SAM3 마스크의 폭 과소추정(−22%) 교정.

frozen SAM3 위에 경량 refiner([gray, sam3_mask] → refined mask)를 **실데이터**(CrackSeg9k)로
학습. 손실 = soft-Dice + λ·면적 보정항(|Σp−Σg|/Σg — 폭 과소추정을 직접 벌점).

엄밀성 (RIGOR_AUDIT 반영):
  - cross-source: TEST_SOURCES(cfd, cracktree200, deepcrack)는 학습에서 통째 제외
  - 설정/조기종료는 train-source val로만, 보고는 test 소스만
  - 체크포인트 저장, 결과 JSON, 폭 GT는 마스크 유래임을 명시

실행:
  python experiments/m2_refiner.py --precompute   # SAM3 마스크 캐시 (GPU, ~10분)
  python experiments/m2_refiner.py --train --epochs 30 --save checkpoints/m2_refiner.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_bench import find_pairs  # noqa: E402

SZ = 256
TEST_SOURCES = {"cfd", "cracktree200", "deepcrack"}   # 통째 홀드아웃 (thin 난제 포함)
CACHE = Path("datasets/m2_cache")


def load_gray_gt(img_p, mask_p):
    from PIL import Image
    g = np.array(Image.open(img_p).convert("L").resize((SZ, SZ)), np.float32) / 255.0
    gt = np.array(Image.open(mask_p).convert("L").resize((SZ, SZ))) > 127
    return g, gt


def build_splits(n_train=1200, n_val=150, n_test_per_src=100, seed=0):
    rng = np.random.default_rng(seed)
    pairs = find_pairs(Path("datasets/crackseg9k"))
    by_src = defaultdict(list)
    for p in pairs:
        by_src[p[2]].append(p)
    train_pool, test_pool = [], []
    for s, v in by_src.items():
        if s == "noncrack":
            continue
        (test_pool if s in TEST_SOURCES else train_pool).extend(
            [v[i] for i in rng.permutation(len(v))[: (n_test_per_src if s in TEST_SOURCES else 10**9)]])
    rng.shuffle(train_pool)
    return train_pool[:n_train], train_pool[n_train:n_train + n_val], test_pool


def precompute(split_name, items):
    """SAM3 'crack' 마스크 캐시 (crack-only 필터 포함)."""
    from gaugeanything.segmenters import segment_sam3
    imgs, sams, gts, srcs = [], [], [], []
    t0 = time.time()
    for k, (ip, mp, src) in enumerate(items):
        g, gt = load_gray_gt(ip, mp)
        if gt.sum() < 30:
            continue
        rgb = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
        insts = segment_sam3(rgb, "crack", threshold=0.4)
        sam = np.any([i.mask for i in insts], axis=0) if insts else np.zeros((SZ, SZ), bool)
        imgs.append((g * 255).astype(np.uint8)); sams.append(sam); gts.append(gt); srcs.append(src)
        if (k + 1) % 200 == 0:
            print(f"  {split_name}: {k+1}/{len(items)} ({time.time()-t0:.0f}s)")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{split_name}.npz",
                        imgs=np.stack(imgs), sams=np.stack(sams),
                        gts=np.stack(gts), srcs=np.array(srcs))
    print(f"  저장: {split_name}.npz  n={len(imgs)}")


def width_of(mask):
    from gaugeanything.geometry import measure_thin
    if mask.sum() < 20:
        return 0.0
    return measure_thin(mask).width_mean


def evaluate(imgs, sams, gts, srcs, model=None, dev="cuda"):
    """raw SAM3 vs refined: crack mIoU + 폭 상대오차(마스크 유래 GT)."""
    import torch
    res = {"raw": defaultdict(list), "refined": defaultdict(list)}
    for i in range(len(imgs)):
        g = imgs[i].astype(np.float32) / 255.0
        sam, gt = sams[i].astype(bool), gts[i].astype(bool)
        w_gt = width_of(gt)
        preds = {"raw": sam}
        if model is not None:
            with torch.no_grad():
                x = torch.tensor(np.stack([g, sam.astype(np.float32)])[None], device=dev)
                preds["refined"] = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= 0.5)
        for name, p in preds.items():
            inter, union = (p & gt).sum(), (p | gt).sum()
            res[name]["iou"].append(inter / union if union else 0.0)
            if w_gt > 0:
                res[name]["wrel"].append(abs(width_of(p) - w_gt) / w_gt)
                res[name]["wbias"].append((width_of(p) - w_gt) / w_gt)
    out = {}
    for name, d in res.items():
        if d["iou"]:
            out[name] = {"mIoU": round(float(np.mean(d["iou"])), 4),
                         "width_rel_err": round(float(np.mean(d["wrel"])), 4),
                         "width_bias": round(float(np.mean(d["wbias"])), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lam-area", type=float, default=0.3)
    ap.add_argument("--save", default="checkpoints/m2_refiner.pt")
    ap.add_argument("--gallery", default=None)
    args = ap.parse_args()

    if args.precompute:
        tr, va, te = build_splits()
        print(f"분할: train {len(tr)} / val {len(va)} / test {len(te)} (test 소스: {sorted(TEST_SOURCES)})")
        for name, items in [("train", tr), ("val", va), ("test", te)]:
            precompute(name, items)
        return 0

    if not args.train:
        print("--precompute 또는 --train 지정"); return 1

    import torch
    import torch.nn.functional as F
    from experiments.draem_uneven import build_unet
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = {k: np.load(CACHE / f"{k}.npz", allow_pickle=True) for k in ("train", "val", "test")}
    print(f"캐시 로드: train {len(D['train']['imgs'])} / val {len(D['val']['imgs'])} / test {len(D['test']['imgs'])}")

    model = build_unet(2, 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xg = torch.tensor(D["train"]["imgs"].astype(np.float32) / 255.0, device=dev)
    Xs = torch.tensor(D["train"]["sams"].astype(np.float32), device=dev)
    Yg = torch.tensor(D["train"]["gts"].astype(np.float32), device=dev)
    n, bs = len(Xg), 16

    def softdice(p, g):
        num = 2 * (p * g).sum((1, 2, 3)) + 1
        den = p.sum((1, 2, 3)) + g.sum((1, 2, 3)) + 1
        return (1 - num / den).mean()

    print(f"학습 {args.epochs}ep (λ_area={args.lam_area}, params={sum(p.numel() for p in model.parameters()):,})")
    best_val, best_state = 1e9, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = torch.stack([Xg[idx], Xs[idx]], 1)
            y = Yg[idx].unsqueeze(1)
            opt.zero_grad()
            p = torch.sigmoid(model(x))
            area = ((p.sum((1, 2, 3)) - y.sum((1, 2, 3))).abs() / (y.sum((1, 2, 3)) + 1)).mean()
            loss = softdice(p, y) + args.lam_area * area
            loss.backward(); opt.step()
        # val (train-source) — 조기 선택은 여기서만
        model.eval()
        v = evaluate(D["val"]["imgs"], D["val"]["sams"], D["val"]["gts"], D["val"]["srcs"], model, dev)
        score = v["refined"]["width_rel_err"] + (1 - v["refined"]["mIoU"])
        if score < best_val:
            best_val, best_state = score, {k: t.clone() for k, t in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:2d}  val refined mIoU={v['refined']['mIoU']:.3f} "
                  f"wrel={v['refined']['width_rel_err']:.3f} bias={v['refined']['width_bias']:+.3f}")
    model.load_state_dict(best_state)
    print(f"학습 완료 {time.time()-t0:.0f}s (best val 기준 체크포인트)")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"저장: {args.save}")

    # --- 공식 평가: 홀드아웃 소스만 ---
    model.eval()
    te = evaluate(D["test"]["imgs"], D["test"]["sams"], D["test"]["gts"], D["test"]["srcs"], model, dev)
    # 소스별
    per_src = {}
    for s in sorted(set(D["test"]["srcs"])):
        m = D["test"]["srcs"] == s
        per_src[s] = evaluate(D["test"]["imgs"][m], D["test"]["sams"][m],
                              D["test"]["gts"][m], D["test"]["srcs"][m], model, dev)
    print(f"\n{'='*70}")
    print(f"공식 (홀드아웃 소스 {sorted(TEST_SOURCES)}, n={len(D['test']['imgs'])}):")
    for name in ("raw", "refined"):
        r = te[name]
        print(f"  {name:8s} mIoU={r['mIoU']:.3f}  width_rel={r['width_rel_err']:.3f}  bias={r['width_bias']:+.3f}")
    print(f"{'='*70}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/m2_refiner.json").write_text(json.dumps(
        {"test_overall": te, "test_per_source": per_src,
         "note": "폭 GT는 마스크 유래(실 mm 아님). test 소스는 학습 미노출."},
        indent=2, ensure_ascii=False))
    print("결과 저장: experiments/results/m2_refiner.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
