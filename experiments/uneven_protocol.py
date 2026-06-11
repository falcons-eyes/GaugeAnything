"""Uneven 평가 프로토콜 — test-set 최적화(A1) 교정판.

MT_Uneven 사용가능 쌍을 고정 시드로 val/test 분할.
  - 고전: {detrend × smooth} 4개 설정을 val에서 선택 → test 보고
  - DRAEM(체크포인트 로드): {disc only, disc+recon-err 앙상블} val 선택 → test 보고
  - SAM3 binary: 설정 없음 → test 직접 (참고치)
공식 수치는 이 스크립트의 test 컬럼만 사용한다.

실행: python experiments/uneven_protocol.py --ckpt checkpoints/draem_uneven.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.draem_uneven import SZ, build_unet, mt_pairs  # noqa: E402
from gaugeanything.soft import mura_severity  # noqa: E402


def auc_of(score, gt):
    from sklearn.metrics import roc_auc_score
    y = gt.ravel().astype(int)
    if y.sum() in (0, y.size):
        return None
    return float(roc_auc_score(y, score.ravel().astype(np.float32)))


def load_pair_sz(ip, mp):
    from PIL import Image
    g = np.array(Image.open(ip).convert("L").resize((SZ, SZ)), np.float32) / 255.0
    gt = np.array(Image.open(mp).convert("L").resize((SZ, SZ))) > 127
    return g, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/draem_uneven.pt")
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args()

    pairs = [(ip, mp) for ip, mp, _ in mt_pairs("Uneven")]
    usable = []
    for ip, mp in pairs:
        g, gt = load_pair_sz(ip, mp)
        if 30 <= gt.sum() < gt.size - 30:
            usable.append((g, gt))
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(usable))
    half = len(usable) // 2
    val = [usable[i] for i in idx[:half]]
    test = [usable[i] for i in idx[half:]]
    print(f"=== Uneven val/test 프로토콜 ===  usable={len(usable)} → val {len(val)} / test {len(test)}")

    def eval_set(samples, score_fn):
        aucs = [a for g, gt in samples if (a := auc_of(score_fn(g), gt)) is not None]
        return float(np.mean(aucs)) if aucs else 0.0

    out = {"split_seed": args.split_seed, "n_val": len(val), "n_test": len(test)}

    # --- 고전: val에서 설정 선택 ---
    print("\n[고전 조명잔차] val 설정 스윕:")
    best = (None, -1)
    for det in (False, True):
        for sm in (0.0, 9.0):
            def fn(g, det=det, sm=sm):
                # mura_severity는 uint8 가정 없음 — float 그대로
                return mura_severity((g * 255).astype(np.float32), order=2,
                                     smooth=sm, detrend_cols=det)["soft_map"]
            v = eval_set(val, fn)
            print(f"  detrend={det!s:5} smooth={sm:>3}: val AUC {v:.4f}")
            if v > best[1]:
                best = ((det, sm), v)
    (det, sm), vbest = best
    test_classical = eval_set(test, lambda g: mura_severity(
        (g * 255).astype(np.float32), order=2, smooth=sm, detrend_cols=det)["soft_map"])
    print(f"  → 선택: detrend={det}, smooth={sm} (val {vbest:.4f}) → **test {test_classical:.4f}**")
    out["classical"] = {"config": {"detrend": det, "smooth": sm},
                        "val": round(vbest, 4), "test": round(test_classical, 4)}

    # --- DRAEM: 체크포인트 로드, 앙상블 여부 val 선택 ---
    if Path(args.ckpt).exists():
        import torch
        from scipy.ndimage import gaussian_filter
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(args.ckpt, map_location=dev)
        recon = build_unet(1, 1).to(dev); recon.load_state_dict(ck["recon"]); recon.eval()
        disc = build_unet(2, 1).to(dev); disc.load_state_dict(ck["disc"]); disc.eval()

        def draem_score(g, ensemble):
            with torch.no_grad():
                x = torch.tensor(g, device=dev)[None, None]
                rec = recon(x)
                seg = torch.sigmoid(disc(torch.cat([x, rec], 1)))[0, 0].cpu().numpy()
                if not ensemble:
                    return seg
                rerr = gaussian_filter(np.abs(g - rec[0, 0].cpu().numpy()), 6)
                rerr = (rerr - rerr.min()) / (np.ptp(rerr) + 1e-6)
                return 0.5 * seg + 0.5 * rerr

        print("\n[DRAEM-lite] val 설정 스윕:")
        bestd = (None, -1)
        for ens in (False, True):
            v = eval_set(val, lambda g, e=ens: draem_score(g, e))
            print(f"  ensemble={ens!s:5}: val AUC {v:.4f}")
            if v > bestd[1]:
                bestd = (ens, v)
        ens, vb = bestd
        test_draem = eval_set(test, lambda g: draem_score(g, ens))
        print(f"  → 선택: ensemble={ens} (val {vb:.4f}) → **test {test_draem:.4f}**")
        out["draem"] = {"config": {"ensemble": ens}, "val": round(vb, 4),
                        "test": round(test_draem, 4)}
    else:
        print(f"\n[DRAEM] 체크포인트 없음({args.ckpt}) — 건너뜀")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/uneven_protocol.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}\n공식 수치(test only): "
          + " · ".join(f"{k} {v['test']}" for k, v in out.items() if isinstance(v, dict) and "test" in v))
    print("결과 저장: experiments/results/uneven_protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
