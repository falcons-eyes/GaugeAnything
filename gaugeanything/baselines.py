"""고전 크랙 분할 베이스라인 — Gauge-Bench의 기준선.

베이스라인 사다리 원칙 (GAP_ANALYSIS §4): 딥러닝이 50년 도메인 지식(국소 임계,
vesselness 필터)을 못 이기면 의미 없다. SAM3 평가 전에 이 기준선부터 세운다.

공통 인터페이스: fn(gray uint8 [H,W]) -> bool mask [H,W]
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_gray(image: np.ndarray) -> np.ndarray:
    g = image if image.ndim == 2 else image.mean(axis=2)
    return g.astype(np.uint8) if g.dtype != np.uint8 else g


def _clean(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    """소형 노이즈 성분 제거."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_size
    return keep[labeled]


def seg_otsu(image: np.ndarray) -> np.ndarray:
    """전역 Otsu — 가장 소박한 기준선."""
    from skimage.filters import threshold_otsu
    g = _to_gray(image)
    return _clean(g <= threshold_otsu(g))


def seg_adaptive(image: np.ndarray, block: int = 51, c: int = 10) -> np.ndarray:
    """국소 평균 대비 임계 — 조명 변화에 강건한 고전 기법."""
    g = _to_gray(image).astype(np.float32)
    local_mean = ndimage.uniform_filter(g, block)
    return _clean(g < local_mean - c)


def seg_frangi(image: np.ndarray, sigmas=(1, 2, 3, 4)) -> np.ndarray:
    """Frangi vesselness — 관상(thin tubular) 구조 전용 필터. 크랙의 고전 SOTA 계열.
    크랙은 어두운 능선이므로 반전 입력."""
    from skimage.filters import frangi, threshold_otsu
    g = _to_gray(image).astype(np.float32) / 255.0
    resp = frangi(1.0 - g, sigmas=sigmas, black_ridges=False)
    if resp.max() <= 0:
        return np.zeros_like(g, bool)
    resp = resp / resp.max()
    return _clean(resp > max(threshold_otsu(resp), 0.05))


def seg_blackhat(image: np.ndarray, size: int = 15) -> np.ndarray:
    """모폴로지 black-hat — 배경보다 어두운 thin 구조 강조."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import black_tophat, disk
    g = _to_gray(image)
    resp = black_tophat(g, disk(size))
    if resp.max() == 0:
        return np.zeros_like(g, bool)
    return _clean(resp > threshold_otsu(resp))


BASELINES = {
    "otsu": seg_otsu,
    "adaptive": seg_adaptive,
    "frangi": seg_frangi,
    "blackhat": seg_blackhat,
}
