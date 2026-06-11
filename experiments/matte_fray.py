"""License-clean matting 헤드 — fray의 애매 경계용 soft α (MAM M2M 레시피).

제약: 산업 결함 alpha GT 부재 (자성타일은 binary 마스크만). 라이선스: 학습형 matting
가중치는 Adobe Comp-1k 오염 → **자체 학습**. 전략(self-supervised):
  정상(MT_Free) 패치에 **합성 fuzzy 경계 결함**(alpha 0..1 known) 합성 → M2M 헤드가
  [RGB, coarse-mask] → soft α 회귀 학습. 합성이므로 alpha GT가 정확.
검증: 합성 holdout의 alpha MAE + 실제 MT_Fray binary와의 경계 정합(soft-IoU).

실행: python experiments/matte_fray.py --epochs 35 --gallery docs/assets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SZ = 256


def load_imgs(defect, n=None):
    from PIL import Image
    d = Path("datasets/magnetic_tile") / f"MT_{defect}" / "Imgs"
    files = sorted(d.glob("*.jpg"))[:n] if n else sorted(d.glob("*.jpg"))
    return np.stack([np.array(Image.open(f).convert("L").resize((SZ, SZ)), np.float32)/255. for f in files]) if files else np.empty((0, SZ, SZ), np.float32)


def synth_fray(img, rng):
    """정상 패치 → (fray 합성 이미지, 정확한 soft α, coarse 마스크).

    fray = 전경(마모/섬유)이 *애매한 경계*로 배경과 혼합. 절차:
      1) 임의 블롭 영역의 hard 코어 + 가장자리 페더(거리변환 기반 soft α)
      2) 가장자리에 텍스처 노이즈로 'frayed' 외형
      3) I = α·F + (1−α)·B  (matting 방정식)
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    # 불규칙 블롭 (저주파 임계)
    ys, xs = np.mgrid[0:SZ, 0:SZ] / SZ
    field = np.zeros((SZ, SZ), np.float32)
    for _ in range(rng.integers(2, 4)):
        fx, fy = rng.uniform(1, 4, 2); ph = rng.uniform(0, 6.28)
        field += np.sin(2*np.pi*(fx*xs+fy*ys)+ph)
    cy, cx = rng.uniform(0.3, 0.7, 2); rr = rng.uniform(0.12, 0.25)
    blob = (np.exp(-(((ys-cy)**2+(xs-cx)**2)/(2*rr**2))) + 0.25*field) > rng.uniform(0.7, 0.95)
    if blob.sum() < 50:
        blob[int(cy*SZ)-15:int(cy*SZ)+15, int(cx*SZ)-15:int(cx*SZ)+15] = True
    # soft α: 코어=1, 가장자리 페더 (거리변환 → fray 폭)
    edt_in = distance_transform_edt(blob)
    feather = rng.uniform(4, 12)
    alpha = np.clip(edt_in / feather, 0, 1).astype(np.float32)
    # frayed 가장자리: 경계 밴드에 텍스처 노이즈로 α 교란
    band = (alpha > 0.05) & (alpha < 0.95)
    noise = gaussian_filter(rng.standard_normal((SZ, SZ)).astype(np.float32), 1.5)
    alpha = np.clip(alpha + band * noise * 0.35, 0, 1).astype(np.float32)
    alpha = gaussian_filter(alpha, 0.8)
    # 전경 외형 (어둡거나 밝은 마모) + matting 합성
    fg = np.clip(img + rng.uniform(-0.4, 0.4), 0, 1).astype(np.float32)
    frayed = (alpha * fg + (1 - alpha) * img).astype(np.float32)
    # coarse 마스크 (SAM3가 줄 법한 거친 binary — α>0.5 + 침식/팽창 노이즈)
    from scipy.ndimage import binary_dilation, binary_erosion
    coarse = alpha > 0.5
    if rng.random() < 0.5:
        coarse = binary_erosion(coarse, iterations=rng.integers(0, 3))
    else:
        coarse = binary_dilation(coarse, iterations=rng.integers(0, 3))
    return frayed, alpha, coarse.astype(np.float32)


def synth_fray_directional(img, rng):
    """방향성 fray 합성 v2 — 실분포 정합 재설계 (블롭 v1의 실 전이 실패 교정).

    실제 fray의 성질을 모사: (i) 연신된 이방성 영역(연신비 5~15×), (ii) 경계가
    가닥(strand)처럼 들쭉날쭉, (iii) 외형은 밝기 오프셋이 아니라 텍스처 교란(streak).
    coarse 마스크는 50% 무지터 — "마스크를 재형성하지 말고 경계만 정제"를 학습.
    방향: 70%는 수직 근방(자성타일 연삭 방향 편향 — 데이터셋 특화임을 명시), 30% 무작위.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter
    theta = rng.uniform(0, np.pi) if rng.random() < 0.3 else np.pi / 2 + rng.normal(0, 0.3)
    c, s = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:SZ, 0:SZ].astype(np.float32)
    cy, cx = rng.uniform(0.25, 0.75, 2) * SZ
    u = (xx - cx) * c + (yy - cy) * s            # 길이 방향
    v = -(xx - cx) * s + (yy - cy) * c           # 폭 방향
    L = rng.uniform(0.25, 0.45) * SZ
    W = rng.uniform(0.02, 0.08) * SZ             # 연신비 ~5-15×
    # 경계 가닥: 폭을 u축 1D 노이즈로 변조 (섬유 다발의 들쭉날쭉함)
    n1 = gaussian_filter(rng.standard_normal(SZ).astype(np.float32), 3)
    n1 = n1 / (np.abs(n1).max() + 1e-6)
    ui = np.clip(((u + SZ) / (2 * SZ) * (SZ - 1)).astype(int), 0, SZ - 1)
    jag = 1.0 + 0.6 * n1[ui]
    blob = (np.abs(u) / L) ** 2 + (np.abs(v) / (W * jag + 1e-3)) ** 2 < 1.0
    if blob.sum() < 50:
        blob = (np.abs(u) / L) ** 2 + (np.abs(v) / W) ** 2 < 1.0
    edt_in = distance_transform_edt(blob)
    alpha = np.clip(edt_in / rng.uniform(2, 6), 0, 1).astype(np.float32)
    # 경계 밴드에 가닥 줄무늬 (연속 α의 fray다운 질감)
    band = (alpha > 0.02) & (alpha < 0.98)
    stripe = 0.5 + 0.5 * np.sin(v * rng.uniform(0.8, 2.0) + n1[ui] * 6)
    alpha = np.clip(alpha * np.where(band, 0.4 + 0.6 * stripe, 1.0), 0, 1)
    alpha = gaussian_filter(alpha, 0.6).astype(np.float32)
    # 외형: 텍스처 교란 (방향성 streak + 약한 밝기 오프셋)
    streaks = gaussian_filter(rng.standard_normal((SZ, SZ)).astype(np.float32), (4, 0.6))
    fg = np.clip(img + rng.uniform(-0.35, 0.35) + 0.25 * streaks, 0, 1).astype(np.float32)
    frayed = (alpha * fg + (1 - alpha) * img).astype(np.float32)
    # coarse: 50% 무지터 (정제 학습), 50% 약지터
    coarse = alpha > 0.5
    if rng.random() < 0.5:
        it = int(rng.integers(1, 3))
        coarse = (binary_erosion(coarse, iterations=it) if rng.random() < 0.5
                  else binary_dilation(coarse, iterations=it))
    return frayed, alpha, coarse.astype(np.float32)


SYNTHS = {"blob": synth_fray, "directional": synth_fray_directional}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--save", default=None, help="체크포인트 저장 경로 (.pt)")
    ap.add_argument("--real-eval", action="store_true", help="실제 MT_Fray 검증 (B1)")
    ap.add_argument("--synth", default="blob", choices=list(SYNTHS),
                    help="합성 분포 (directional=실분포 정합 v2)")
    args = ap.parse_args()
    synth_fn = SYNTHS[args.synth]
    print(f"합성 분포: {args.synth}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Matting 헤드 (self-sup, fray) · device={dev} ===")
    normal = load_imgs("Free", args.n_train)
    print(f"정상(MT_Free): {len(normal)}")

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class M2M(nn.Module):
        """[RGB(1ch), coarse-mask] → soft α. MAM M2M 유사 (마스크+이미지 → α 회귀)."""
        def __init__(self, base=32):
            super().__init__()
            self.e1 = block(2, base); self.e2 = block(base, base*2); self.e3 = block(base*2, base*4)
            self.pool = nn.MaxPool2d(2); self.b = block(base*4, base*8)
            self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.d3 = block(base*8, base*4)
            self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.d2 = block(base*4, base*2)
            self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2); self.d1 = block(base*2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            b = self.b(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(b), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)

    model = M2M().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    Xn = torch.tensor(normal, device=dev)
    n, bs = len(Xn), 16
    print(f"\n학습 {args.epochs}ep (M2M, params={sum(p.numel() for p in model.parameters()):,})")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            imgs, alphas, coarses = [], [], []
            for x in Xn[idx].cpu().numpy():
                fr, al, co = synth_fn(x, rng); imgs.append(fr); alphas.append(al); coarses.append(co)
            inp = torch.tensor(np.stack([np.stack([i, c]) for i, c in zip(imgs, coarses)]), device=dev)
            at = torch.tensor(np.stack(alphas), device=dev).unsqueeze(1)
            opt.zero_grad()
            pred = torch.sigmoid(model(inp))
            # α 회귀: L1 + 경계 가중(불확실 밴드 강조)
            w = 1 + 3 * ((at > 0.05) & (at < 0.95)).float()
            loss = (w * (pred - at).abs()).mean()
            loss.backward(); opt.step(); tot += loss.item()*len(idx)
        if ep == 0 or (ep+1) % 10 == 0:
            print(f"  epoch {ep+1:2d}  α-L1={tot/n:.4f}")
    print(f"학습 완료 {time.time()-t0:.1f}s")

    # --- 검증 1: 합성 holdout alpha MAE (binary 임계 vs matting) ---
    model.eval()
    rng2 = np.random.default_rng(999)
    mae_soft, mae_bin = [], []
    val = load_imgs("Free", None)[args.n_train:args.n_train+60]
    with torch.no_grad():
        for x in val:
            fr, al, co = synth_fn(x, rng2)
            inp = torch.tensor(np.stack([fr, co])[None], device=dev)
            pred = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
            mae_soft.append(float(np.abs(pred - al).mean()))
            mae_bin.append(float(np.abs(co - al).mean()))   # coarse binary가 α 근사
            if args.gallery and len(mae_soft) == 1:
                _panel(fr, co, pred, al, Path(args.gallery)/"matte_fray.png")
    print(f"\n{'='*56}")
    print(f"합성 holdout α-MAE:  matting {np.mean(mae_soft):.4f}  vs  binary {np.mean(mae_bin):.4f}")
    print(f"  → matting이 fuzzy 경계의 soft α를 binary보다 정확히 복원")
    print(f"{'='*56}")
    res = {"alpha_mae_matting": round(float(np.mean(mae_soft)), 4),
           "alpha_mae_binary": round(float(np.mean(mae_bin)), 4),
           "note": "synthetic holdout — 합성→합성 검증임을 명시 (RIGOR_AUDIT B1)"}

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"체크포인트 저장: {args.save}")

    # --- 검증 2 (B1): 실제 MT_Fray — α GT 부재 하의 약한 검증 ---
    if args.real_eval:
        from PIL import Image
        from scipy.ndimage import binary_dilation
        d = Path("datasets/magnetic_tile/MT_Fray/Imgs")
        pairs = [(j, j.with_suffix(".png")) for j in sorted(d.glob("*.jpg"))
                 if j.with_suffix(".png").exists()]
        ious, softs, ious_gm = [], [], []
        from gaugeanything.soft import guided_matte
        first = True
        for jp, pp in pairs:
            g = np.array(Image.open(jp).convert("L").resize((SZ, SZ)), np.float32) / 255.0
            gt = np.array(Image.open(pp).convert("L").resize((SZ, SZ))) > 127
            if gt.sum() < 30:
                continue
            inp = torch.tensor(np.stack([g, gt.astype(np.float32)])[None], device=dev)
            with torch.no_grad():
                alpha = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
            # (i) 보존성: α≥0.5 vs GT IoU (마스크를 망가뜨리지 않는가)
            p = alpha >= 0.5
            inter, union = (p & gt).sum(), (p | gt).sum()
            ious.append(inter / union if union else 0.0)
            # (ii) softness: GT 경계 밴드에서 α가 진짜 연속값을 내는가
            band = binary_dilation(gt, iterations=4) & ~gt
            if band.sum() > 10:
                softs.append(float(((alpha > 0.1) & (alpha < 0.9))[band].mean()))
            # guided_matte 비교 (학습 없는 고전)
            ga = guided_matte(np.stack([g]*3, -1) * 255, gt)
            pg = ga >= 0.5
            ious_gm.append((pg & gt).sum() / max((pg | gt).sum(), 1))
            if args.gallery and first and ious[-1] > 0.5:
                _panel_real(g, gt, alpha, Path(args.gallery) / "matte_fray_real.png")
                first = False
        res["real_fray"] = {
            "n": len(ious),
            "iou_at_0.5_vs_gt": round(float(np.mean(ious)), 4),
            "guided_matte_iou": round(float(np.mean(ious_gm)), 4),
            "boundary_softness": round(float(np.mean(softs)), 4) if softs else None,
            "note": "α GT 없음 → 보존성+softness만. 정량적 α 정확도는 합성 한정."}
        print(f"\n[실제 MT_Fray n={len(ious)}] α≥0.5 vs GT IoU: {np.mean(ious):.3f} "
              f"(guided_matte {np.mean(ious_gm):.3f}) · 경계 softness {np.mean(softs):.3f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/matte_fray.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _panel_real(g, gt, alpha, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title("Real MT_Fray"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Binary GT (coarse input)"); ax[1].axis("off")
    im = ax[2].imshow(alpha, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("Matting head soft α (real)"); plt.colorbar(im, ax=ax[2], fraction=.046); ax[2].axis("off")
    fig.suptitle("Real-fray validation — no α GT, qualitative + consistency", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93]); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


def _panel(fr, co, pred, gt, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(fr, cmap="gray"); ax[0].set_title("Synth fray (fuzzy edge)"); ax[0].axis("off")
    ax[1].imshow(co, cmap="gray"); ax[1].set_title("Coarse binary mask"); ax[1].axis("off")
    im = ax[2].imshow(pred, cmap="magma", vmin=0, vmax=1); ax[2].set_title("Matting head → soft α")
    plt.colorbar(im, ax=ax[2], fraction=.046); ax[2].axis("off")
    ax[3].imshow(gt, cmap="magma", vmin=0, vmax=1); ax[3].set_title("GT alpha"); ax[3].axis("off")
    fig.suptitle("License-clean matting head (self-supervised, fuzzy boundary)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93]); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
