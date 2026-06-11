"""DRAEM-lite (저주파 합성) — uneven/mura 학습형 검출. 목표: 고전 0.683 돌파.

DRAEM 구조(reconstructive + discriminative)를 가져오되, 핵심 수정:
  합성 이상 = Perlin 패치(경계 있는 구조) → **저주파 밝기장**(경계 없는 점진 불균일).
정상(MT_Free) 텍스처를 학습 → 매끄러운 밝기 편차를 이상으로 판별.

실행: python experiments/draem_uneven.py --epochs 40 --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gauge_multidomain import load_pair, mt_pairs  # noqa: E402

SZ = 256


def load_imgs(defect: str, n: int | None = None):
    from PIL import Image
    d = Path("datasets/magnetic_tile") / f"MT_{defect}" / "Imgs"
    files = sorted(d.glob("*.jpg"))
    if n:
        files = files[:n]
    out = []
    for f in files:
        g = np.array(Image.open(f).convert("L").resize((SZ, SZ)), np.float32) / 255.0
        out.append(g)
    return np.stack(out) if out else np.empty((0, SZ, SZ), np.float32)


def lowfreq_field(rng, h=SZ, w=SZ, k=3):
    """매끄러운 저주파장 (랜덤 2D 사인 합) → [-1,1] 정규화. 경계 없음."""
    ys, xs = np.mgrid[0:h, 0:w] / max(h, w)
    f = np.zeros((h, w), np.float32)
    for _ in range(k):
        fx, fy = rng.uniform(0.5, 3, 2)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.4, 1.0)
        f += amp * np.sin(2 * np.pi * (fx * xs + fy * ys) + ph)
    f -= f.mean()
    return (f / (np.abs(f).max() + 1e-6)).astype(np.float32)


def synth_anomaly(img, rng):
    """정상 → (이상 이미지, soft 마스크). 저주파 밝기장 곱셈."""
    F = lowfreq_field(rng)
    amp = rng.uniform(0.15, 0.5)
    # 대부분 국소화 (MT_Uneven 결함은 국소적) — 가우시안 블롭 윈도우
    if rng.random() < 0.85:
        cy, cx = rng.uniform(0.2, 0.8, 2) * SZ
        r = rng.uniform(0.12, 0.35) * SZ
        ys, xs = np.mgrid[0:SZ, 0:SZ]
        win = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * r ** 2)).astype(np.float32)
        F = F * win
    anom = np.clip(img * (1 + amp * F), 0, 1).astype(np.float32)
    mask = np.clip(np.abs(amp * F) / (amp * 0.5), 0, 1).astype(np.float32)  # soft 타깃
    return anom, mask


def build_unet(ic, oc, base=32):
    """모듈 레벨 UNet 팩토리 — 프로토콜 스크립트에서 체크포인트 로드용."""
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = block(ic, base); self.e2 = block(base, base*2); self.e3 = block(base*2, base*4)
            self.pool = nn.MaxPool2d(2)
            self.b = block(base*4, base*8)
            self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.d3 = block(base*8, base*4)
            self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.d2 = block(base*4, base*2)
            self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2); self.d1 = block(base*2, base)
            self.out = nn.Conv2d(base, oc, 1)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            b = self.b(self.pool(e3))
            d3 = self.d3(torch.cat([self.up3(b), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)

    return UNet()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--save", default=None, help="체크포인트 저장 경로 (.pt)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DRAEM-lite (저주파 합성) · device={dev} ===")

    normal = load_imgs("Free", args.n_train)
    print(f"정상(MT_Free) 학습 이미지: {len(normal)}")

    recon = build_unet(1, 1).to(dev)  # 이상 이미지 → 정상 복원
    disc = build_unet(2, 1).to(dev)   # [이상, 복원] → 이상 마스크
    opt = torch.optim.Adam(list(recon.parameters()) + list(disc.parameters()), 1e-3)
    rng = np.random.default_rng(0)
    Xn = torch.tensor(normal, device=dev).unsqueeze(1)
    n, bs = len(Xn), 16

    print(f"\n학습 {args.epochs}ep (recon+disc, params={sum(p.numel() for p in list(recon.parameters())+list(disc.parameters())):,})")
    t0 = time.time()
    for ep in range(args.epochs):
        recon.train(); disc.train(); perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; xb = Xn[idx]
            # 배치별 합성 이상
            anoms, masks = [], []
            for x in xb.cpu().numpy()[:, 0]:
                a, m = synth_anomaly(x, rng); anoms.append(a); masks.append(m)
            xa = torch.tensor(np.stack(anoms), device=dev).unsqueeze(1)
            mt = torch.tensor(np.stack(masks), device=dev).unsqueeze(1)
            opt.zero_grad()
            rec = recon(xa)
            seg = disc(torch.cat([xa, rec], 1))
            loss = F.mse_loss(rec, xb) + F.binary_cross_entropy_with_logits(seg, mt)
            loss.backward(); opt.step(); tot += loss.item()*len(idx)
        if ep == 0 or (ep+1) % 10 == 0:
            print(f"  epoch {ep+1:2d}  loss={tot/n:.4f}")
    print(f"학습 완료 {time.time()-t0:.1f}s")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"recon": recon.state_dict(), "disc": disc.state_dict()}, args.save)
        print(f"체크포인트 저장: {args.save}")
    print("주의: 아래 자체 평가는 탐색용(exploratory) — 공식 수치는 uneven_protocol.py(val/test 분리)로.")

    # --- 평가: MT_Uneven AUC ---
    from sklearn.metrics import roc_auc_score
    recon.eval(); disc.eval()
    pairs = mt_pairs("Uneven")
    aucs = []; first = True
    from PIL import Image
    with torch.no_grad():
        for ip, mp, _ in pairs[:60]:
            g = np.array(Image.open(ip).convert("L").resize((SZ, SZ)), np.float32)/255.0
            gt = np.array(Image.open(mp).convert("L").resize((SZ, SZ))) > 127
            if gt.sum() < 20:
                continue
            x = torch.tensor(g, device=dev)[None, None]
            rec = recon(x)
            seg = torch.sigmoid(disc(torch.cat([x, rec], 1)))[0, 0].cpu().numpy()
            # DRAEM 앙상블: 판별맵 + reconstruction error (정상 학습 AE가 못 복원한 영역)
            rerr = np.abs(x[0, 0].cpu().numpy() - rec[0, 0].cpu().numpy())
            from scipy.ndimage import gaussian_filter
            rerr = gaussian_filter(rerr, 6)
            rerr = (rerr - rerr.min()) / (np.ptp(rerr) + 1e-6)
            seg = 0.5 * seg + 0.5 * rerr
            y = gt.ravel().astype(int)
            if y.sum() == 0 or y.sum() == y.size:
                continue
            aucs.append(float(roc_auc_score(y, seg.ravel())))
            if args.gallery and first:
                _panel(g, gt, seg, Path(args.gallery)/"draem_uneven.png"); first = False
    auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"\n{'='*56}")
    print(f"DRAEM-lite MT_Uneven AUC: {auc:.4f}  (n={len(aucs)})")
    print(f"  고전 조명잔차 0.683 / SAM3 binary 0.499 대비")
    print(f"{'='*56}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/draem_uneven.json").write_text(
        json.dumps({"auc": round(auc, 4), "n": len(aucs), "baseline_classical": 0.683,
                    "baseline_sam3": 0.499}, indent=2))
    return 0


def _panel(g, gt, seg, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title("Input (MT_Uneven)"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    im = ax[2].imshow(seg, cmap="inferno"); ax[2].set_title("DRAEM-lite anomaly map")
    plt.colorbar(im, ax=ax[2], fraction=.046); ax[2].axis("off")
    fig.suptitle("Learned field anomaly (low-freq synthesis)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93]); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
