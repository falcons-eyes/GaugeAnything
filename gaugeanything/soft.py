"""Soft Inspection — 경계 없는/애매한 결함의 연속 표현 (SOFT_INSPECTION.md).

세 도구:
  (a′) fray  → guided_matte(): SAM3 마스크를 이미지 엣지로 feather → soft α (matting-lite, license-clean)
  (b)  uneven→ illumination_residual()/mura_severity(): 매끄러운 조명장 잔차 = severity field
  측정 → soft_area/soft_iso_length/soft_width/severity_score: α partial-volume 기반
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def _gray(image: np.ndarray) -> np.ndarray:
    g = image if image.ndim == 2 else image.mean(2)
    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# (b) uneven/mura — 조명장 모델 + 잔차
# ---------------------------------------------------------------------------
def fit_poly_surface(gray: np.ndarray, order: int = 2) -> np.ndarray:
    """2D 다항식 surface 적합 (매끄러운 조명/배경 모델). 반환: 적합면."""
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xs / w, ys / h
    # 차수 order까지의 단항식 [1, x, y, x^2, xy, y^2, ...]
    terms = [np.ones_like(xn)]
    for d in range(1, order + 1):
        for i in range(d + 1):
            terms.append((xn ** (d - i)) * (yn ** i))
    A = np.stack([t.ravel() for t in terms], 1)
    coef, *_ = np.linalg.lstsq(A, gray.ravel(), rcond=None)
    return (A @ coef).reshape(h, w).astype(np.float32)


def illumination_residual(image: np.ndarray, order: int = 2, mode: str = "subtract") -> np.ndarray:
    """이미지 − 매끄러운 조명장 = 잔차. mode='divide'면 image/illum−1 (곱셈 조명)."""
    g = _gray(image)
    fit = fit_poly_surface(g, order)
    if mode == "divide":
        return (g / (fit + 1e-3) - 1.0).astype(np.float32)
    return (g - fit).astype(np.float32)


def mura_severity(image: np.ndarray, order: int = 2, smooth: float = 9.0,
                  detrend_cols: bool = False) -> dict:
    """uneven/mura severity (ISO 25178 areal roughness, 조명보정 잔차 기준).

    mura는 저주파장 → 잔차를 평활화해 고주파 표면 텍스처를 억제.
    detrend_cols: 방향성(연삭) 텍스처를 컬럼 중앙값으로 제거. ⚠️ 데이터셋 특화 옵션 —
      세로 방향 실제 결함도 지울 수 있으므로 기본 False (RIGOR_AUDIT C1). 자성타일처럼
      수직 텍스처가 지배적인 표면에서만 명시적으로 켤 것.
    반환: {Sa, Sq, Ssk, Sku, soft_map[0..1], grad_p95}."""
    g = _gray(image)
    if detrend_cols:
        g = g - np.median(g, axis=0, keepdims=True)  # 컬럼별 정규화 (수직 텍스처 억제)
    r = (g - fit_poly_surface(g, order)).astype(np.float32)
    Sa = float(np.mean(np.abs(r)))
    Sq = float(np.sqrt(np.mean(r ** 2)))
    Ssk = float(np.mean(r ** 3) / (Sq ** 3 + 1e-9))
    Sku = float(np.mean(r ** 4) / (Sq ** 4 + 1e-9))
    # soft 이상맵: |잔차| 평활화(저주파 mura 보존, 텍스처 억제) → robust 정규화
    a = np.abs(r)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, smooth)
    hi = np.percentile(a, 99) + 1e-6
    soft = np.clip(a / hi, 0, 1).astype(np.float32)
    gy, gx = np.gradient(r)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return {"Sa": Sa, "Sq": Sq, "Ssk": Ssk, "Sku": Sku,
            "grad_p95": float(np.percentile(grad, 95)), "soft_map": soft}


# ---------------------------------------------------------------------------
# (a′) fray — guided-filter matting (binary 마스크 → soft α)
# ---------------------------------------------------------------------------
def guided_matte(image: np.ndarray, mask: np.ndarray, radius: int = 8,
                 eps: float = 1e-3, feather: int = 2) -> np.ndarray:
    """SAM3 binary 마스크 → soft α. guided filter가 이미지 엣지를 따라 경계를 feather.
    license-clean(고전). 학습형 matting의 PoC 대용."""
    g = _gray(image)
    gn = (g - g.min()) / (np.ptp(g) + 1e-6)
    m = mask.astype(np.float32)
    if feather and _HAS_CV2:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    if _HAS_CV2 and hasattr(cv2, "ximgproc"):
        alpha = cv2.ximgproc.guidedFilter(gn.astype(np.float32), m, radius, eps)
    else:  # 폴백: 가우시안 feather만
        from scipy.ndimage import gaussian_filter
        alpha = gaussian_filter(m, radius / 2)
    return np.clip(alpha, 0, 1).astype(np.float32)


def trimap_from_mask(mask: np.ndarray, k: int = 15, it: int = 5) -> np.ndarray:
    """Matte-Anything auto-trimap: fg=1, unknown=0.5, bg=0 (erode/dilate)."""
    if not _HAS_CV2:
        raise RuntimeError("opencv 필요")
    m = (mask > 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.erode(m, ker, iterations=it)
    bg = cv2.dilate(m, ker, iterations=it)
    tri = np.full(m.shape, 0.5, np.float32)
    tri[bg == 0] = 0.0
    tri[fg == 1] = 1.0
    return tri


# ---------------------------------------------------------------------------
# Soft 측정 (partial-volume 기반)
# ---------------------------------------------------------------------------
def soft_area(alpha: np.ndarray, mm_per_px: float | None = None) -> float:
    """면적 = Σα (경계 partial-volume 복원). mm_per_px 있으면 mm²."""
    s = (mm_per_px or 1.0) ** 2
    return float(alpha.sum() * s)


def soft_iso_length(alpha: np.ndarray, level: float = 0.5, mm_per_px: float | None = None) -> float:
    """α=level iso-contour 길이 (sub-pixel, marching squares)."""
    from skimage import measure
    s = mm_per_px or 1.0
    total = 0.0
    for c in measure.find_contours(alpha, level):
        total += float(np.sum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, 1))))
    return total * s


def soft_width(alpha: np.ndarray, level: float = 0.5, mm_per_px: float | None = None) -> dict:
    """soft 폭 프로파일: 스켈레톤 수직 α 적분 근사. thin 결함용."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    s = mm_per_px or 1.0
    binm = alpha >= level
    if binm.sum() < 20:
        return {"mean": 0.0, "p95": 0.0}
    # soft 폭 근사: 총 α질량 / 스켈레톤 길이 (수직 적분의 평균)
    skel = skeletonize(binm)
    skel_len = max(float(skel.sum()), 1.0)
    mean_w = float(alpha.sum() / skel_len)
    # p95는 EDT 기반 보조
    edt = ndimage.distance_transform_edt(binm)
    p95 = float(np.percentile(2 * edt[skel], 95)) if skel.sum() else 0.0
    return {"mean": mean_w * s, "p95": p95 * s}


def severity_score(alpha: np.ndarray, roi_area: int | None = None) -> dict:
    """확산 결함 severity = extent × intensity (ASTM D610식 로그 전이) → 등급."""
    roi = roi_area or alpha.size
    extent = float(alpha.sum() / max(roi, 1))            # 0..1 coverage
    inside = alpha[alpha > 0.05]
    intensity = float(np.percentile(inside, 90)) if inside.size else 0.0
    # 로그 전이 (ASTM D610은 지수 스케일) → 0..1 점수
    score = float(np.clip(0.5 * np.log1p(extent * 100) / np.log1p(100) + 0.5 * intensity, 0, 1))
    grade = ("Good", "Fair", "Poor", "Severe")[min(int(score * 4), 3)]
    return {"extent": round(extent, 4), "intensity": round(intensity, 3),
            "score": round(score, 3), "grade": grade}


def area_uncertainty(alpha: np.ndarray, mm_per_px: float | None = None) -> float:
    """면적 분산 닫힌형 근사 Var=Σα(1−α)·px⁴ (공간상관 무시, 하한). 반환: std."""
    s = (mm_per_px or 1.0) ** 2
    var = float(np.sum(alpha * (1 - alpha)))
    return float(np.sqrt(var) * s)
