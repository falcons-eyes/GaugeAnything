"""SAM3 transformers API introspection + 스모크 테스트.

신규 모델이라 API를 추측하지 않고 실제 시그니처를 확인한다.
실행: python experiments/sam3_probe.py
"""
from __future__ import annotations

import inspect
import sys

import transformers


def main():
    print("transformers", transformers.__version__)
    sam3 = [n for n in dir(transformers) if "Sam3" in n]
    print("Sam3 클래스:", sam3)

    # 이미지 레벨 PCS = Sam3Processor + Sam3Model (Auto*는 Video 변형으로 잘못 해석됨)
    from transformers import Sam3Model, Sam3Processor
    print("\n[로드] Sam3Processor + Sam3Model (이미지 레벨)...")
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    print("processor:", type(proc).__name__)
    pp = [m for m in dir(proc) if m.startswith("post_process")]
    print("post_process* 메서드:", pp)
    try:
        print("__call__ params:", list(inspect.signature(proc.__call__).parameters)[:14])
    except (ValueError, TypeError):
        print("__call__ params: (introspection 불가)")
    for m in pp:
        try:
            print(f"  {m} params:", list(inspect.signature(getattr(proc, m)).parameters)[:12])
        except (ValueError, TypeError):
            pass

    import torch
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu").eval()
    print("\nmodel:", type(model).__name__)
    print("forward params:", list(inspect.signature(model.forward).parameters)[:14])

    # --- 스모크: 합성 이미지(밝은 배경 + 어두운 사각형) + text='dark square' ---
    import numpy as np
    from PIL import Image
    arr = np.full((480, 640, 3), 220, np.uint8)
    arr[180:300, 260:380] = 30  # 어두운 사각형 1개
    img = Image.fromarray(arr)
    print("\n[스모크] text='square' 추론...")
    try:
        inputs = proc(images=img, text="square", return_tensors="pt").to(model.device)
        print("입력 키:", list(inputs.keys()))
        with torch.no_grad():
            out = model(**inputs)
        print("출력 타입:", type(out).__name__)
        print("출력 속성:", [k for k in dir(out) if not k.startswith("_")][:16])
        # post-process 시도
        for ppm in pp:
            try:
                fn = getattr(proc, ppm)
                res = fn(out, threshold=0.4, target_sizes=[img.size[::-1]])
                print(f"  {ppm} → 타입 {type(res).__name__}, "
                      f"키 {list(res[0].keys()) if isinstance(res, list) and res and hasattr(res[0],'keys') else res}")
            except Exception as e:
                print(f"  {ppm} 실패: {str(e)[:120]}")
    except Exception as e:
        import traceback
        print("추론 실패:", type(e).__name__, str(e)[:200])
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
