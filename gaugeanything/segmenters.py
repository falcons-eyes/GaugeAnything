"""세그멘터 어댑터 — SAM 3(중심) / SAM 2(폴백) / threshold(테스트용).

공통 인터페이스 (Grounded-SAM 합성 계약 + 개념 프롬프트):
    segment(image, prompt) -> list[Instance(mask, score, label)]

SAM 3: 개념 프롬프트(명사구/예시) → 모든 인스턴스 (HF gated — 라이선스 동의 + 로그인 필요)
SAM 2: 포인트/박스 프롬프트 (Apache, 비gated)
threshold: 모델 없는 결정적 폴백 — 계측 코어 검증용
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    mask: np.ndarray      # [H,W] bool
    score: float
    label: str = ""
    bbox: tuple | None = None  # (x1,y1,x2,y2)


def _mask_bbox(m: np.ndarray) -> tuple:
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ---------------------------------------------------------------------------
# threshold — 모델 없는 폴백 (어두운 구조 분할: 크랙 등). 결정적, selftest용.
# ---------------------------------------------------------------------------
def segment_threshold(image: np.ndarray, prompt: str = "", percentile: float | None = None,
                      min_area: int = 50) -> list[Instance]:
    from scipy import ndimage
    gray = image if image.ndim == 2 else image.mean(axis=2)
    if percentile is not None:
        thr = np.percentile(gray, percentile)
    else:
        # Otsu: 어두운 구조가 소수일 때 percentile보다 강건 (배경 병합 방지)
        try:
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(gray)
        except ImportError:
            thr = (float(gray.min()) + float(gray.max())) / 2
    # skimage Otsu 규약: 어두운 클래스는 임계값 '이하' (foreground = > thr)
    binary = gray <= thr
    labeled, n = ndimage.label(binary)
    out = []
    for i in range(1, n + 1):
        m = labeled == i
        if m.sum() < min_area:
            continue
        out.append(Instance(mask=m, score=1.0, label=prompt or "dark_structure",
                            bbox=_mask_bbox(m)))
    return sorted(out, key=lambda x: -x.mask.sum())


# ---------------------------------------------------------------------------
# SAM 3 — 개념 프롬프트 → 전체 인스턴스 (HF transformers, gated 가중치)
# ---------------------------------------------------------------------------
_SAM3 = {}


def segment_sam3(image: np.ndarray, prompt: str, threshold: float = 0.5,
                 mask_threshold: float = 0.5, model_id: str = "facebook/sam3") -> list[Instance]:
    """SAM 3 이미지 PCS: 명사구 → 모든 인스턴스 마스크. 최초 호출 시 모델 로드(캐시).

    검증된 API (sam3_probe.py): Sam3Processor(images, text) → Sam3Model →
    post_process_instance_segmentation(outputs, threshold, mask_threshold, target_sizes)
    → [{scores, boxes, masks}]. (Auto*는 Video 변형이라 사용 금지.)
    """
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor  # transformers>=5.10
    except ImportError as e:
        raise RuntimeError(
            "SAM 3 사용 불가: pip install -U 'transformers>=5.10' torch 후, "
            "https://huggingface.co/facebook/sam3 라이선스 동의 + hf auth login"
        ) from e
    if "model" not in _SAM3:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM3["proc"] = Sam3Processor.from_pretrained(model_id)
        _SAM3["model"] = Sam3Model.from_pretrained(model_id, dtype=torch.bfloat16).to(dev).eval()
        _SAM3["dev"] = dev
    proc, model = _SAM3["proc"], _SAM3["model"]
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image.astype(np.uint8))
    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(_SAM3["dev"])
    import torch
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=mask_threshold,
        target_sizes=[pil.size[::-1]])[0]
    instances = []
    for m, s in zip(res["masks"], res["scores"]):
        mb = np.asarray(m.cpu()).squeeze().astype(bool)
        if mb.sum() == 0:
            continue
        instances.append(Instance(mask=mb, score=float(s), label=prompt, bbox=_mask_bbox(mb)))
    return instances


# ---------------------------------------------------------------------------
# 프롬프트 앙상블 — 동의어 붕괴(prompt brittleness) 대응
# (progress/2026-06-11: "fracture"→0.0, "pit"→0.0 발견에 대한 대응책)
# ---------------------------------------------------------------------------
# 사용자 어휘 → 검증된 프롬프트 셋. 미등록 프롬프트는 자기 자신 + 베이스 확장.
PROMPT_SETS: dict[str, list[str]] = {
    "crack": ["crack", "cracks", "thin dark crack"],
    "fracture": ["crack", "cracks", "fracture"],          # 동의어 → 검증 프롬프트로 구조
    "hole": ["hole", "small dark spot", "blowhole"],
    "pit": ["hole", "small dark spot", "pit"],
    "scratch": ["scratch", "crack", "dark line"],
}


def _mask_iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def segment_sam3_ensemble(image: np.ndarray, prompt: str, threshold: float = 0.4,
                          prompts: list[str] | None = None,
                          dedup_iou: float = 0.5) -> list[Instance]:
    """여러 프롬프트의 인스턴스 합집합 + IoU 중복 제거(최고 score 유지).

    단일 프롬프트가 어휘 붕괴(0.0)해도 셋 내 다른 프롬프트가 구조한다.
    비용: |prompts|배 추론 — 검증된 3개 내외 유지 권장.
    """
    plist = prompts or PROMPT_SETS.get(prompt.lower(), [prompt])
    pool: list[Instance] = []
    for p in plist:
        pool += segment_sam3(image, p, threshold=threshold)
    pool.sort(key=lambda i: -i.score)
    kept: list[Instance] = []
    for inst in pool:
        if all(_mask_iou(inst.mask, k.mask) < dedup_iou for k in kept):
            kept.append(inst)
    return kept


# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------
SEGMENTERS = {
    "threshold": segment_threshold,
    "sam3": segment_sam3,
    "sam3_ensemble": segment_sam3_ensemble,
}


def get_segmenter(name: str):
    if name not in SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r}; available: {list(SEGMENTERS)}")
    return SEGMENTERS[name]
