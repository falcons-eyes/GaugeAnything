"""EDA 메타데이터 가용성 프로브 — SAM3 출력에서 logit/soft-mask 접근 경로 확인.

EDA 본 리포트 전에, 어떤 confidence/logit 메타를 robust하게 뽑을 수 있는지 1회 확인.
실행: python experiments/eda_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    arr = np.full((480, 640, 3), 220, np.uint8)
    arr[180:300, 260:380] = 30
    pil = Image.fromarray(arr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(dev).eval()
    inputs = proc(images=pil, text="square", return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)

    print("=== output 속성 ===")
    keys = [k for k in dir(out) if not k.startswith("_")]
    print(keys)
    for k in keys:
        try:
            v = getattr(out, k)
            if torch.is_tensor(v):
                print(f"  {k}: tensor {tuple(v.shape)} dtype={v.dtype} "
                      f"min={v.float().min():.3f} max={v.float().max():.3f}")
        except Exception:
            pass

    print("\n=== post_process_instance_segmentation 반환 구조 ===")
    res = proc.post_process_instance_segmentation(out, threshold=0.2, mask_threshold=0.5,
                                                  target_sizes=[pil.size[::-1]])[0]
    for k, v in res.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape {tuple(v.shape)} dtype={getattr(v,'dtype',None)}")
            if k == "masks":
                m = v[0] if len(v) else None
                if m is not None:
                    u = np.unique(np.asarray(m.cpu()))[:6]
                    print(f"    mask[0] unique(앞6): {u}  → {'soft' if len(u)>2 else 'binary'}")
        else:
            print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v,'__len__') else '?'} → {v}")

    # 낮은 threshold로 confidence 스펙트럼 가능한지
    res2 = proc.post_process_instance_segmentation(out, threshold=0.0, mask_threshold=0.5,
                                                   target_sizes=[pil.size[::-1]])[0]
    print(f"\nthreshold=0.0 → 인스턴스 {len(res2['scores'])}개, "
          f"score 범위 [{float(min(res2['scores'])):.3f}, {float(max(res2['scores'])):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
