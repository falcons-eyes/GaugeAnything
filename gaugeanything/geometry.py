"""마스크 기하 모듈 — GaugeAnything의 계측 코어.

이진 마스크 → 미터법 치수. 두 기하 클래스:
  - thin (크랙/용접심/와이어): skeleton + EDT → 폭 프로파일, 길이
  - blob (볼트머리/부품/결함): 등가직경, 최소외접사각형 W×H, 면적

폭 추정 원리: 유클리드 거리 변환(EDT)은 각 전경 픽셀에서 배경까지 거리.
스켈레톤(중심축) 위에서 EDT×2 = 국소 폭. 합성 GT로 ±10% 이내 검증 (selftest.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


@dataclass
class GeometryMetrics:
    """단일 인스턴스의 기하 계측 결과. mm_per_px 미지정 시 단위는 px."""

    kind: str                      # "thin" | "blob"
    unit: str                      # "mm" | "px"
    area: float = 0.0
    length: float = 0.0            # thin: 스켈레톤 길이 / blob: 장축
    width_mean: float = 0.0        # thin: 폭 프로파일 평균 / blob: 단축
    width_max: float = 0.0
    width_p95: float = 0.0
    equiv_diameter: float = 0.0    # blob: 등가직경
    width_profile: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    centroid: tuple[float, float] = (0.0, 0.0)  # (row, col) px — 간격 계산용

    def scaled(self) -> dict:
        """요약 dict (JSON 직렬화용)."""
        return {
            "kind": self.kind, "unit": self.unit,
            "area": round(self.area, 3), "length": round(self.length, 3),
            "width_mean": round(self.width_mean, 3), "width_max": round(self.width_max, 3),
            "width_p95": round(self.width_p95, 3),
            "equiv_diameter": round(self.equiv_diameter, 3),
        }


def _aspect_ratio(mask: np.ndarray) -> float:
    """스켈레톤 길이 대비 평균 폭 비율로 thin/blob 자동 판별에 사용."""
    area = float(mask.sum())
    if area == 0:
        return 0.0
    skel = skeletonize(mask)
    skel_len = float(skel.sum())
    if skel_len < 3:
        return 0.0
    mean_w = area / skel_len
    return skel_len / max(mean_w, 1e-6)


def classify_kind(mask: np.ndarray, thin_threshold: float = 8.0) -> str:
    """마스크 형상 자동 분류. 길이/폭 비가 크면 thin (크랙류)."""
    if not _HAS_SKIMAGE:
        return "blob"
    return "thin" if _aspect_ratio(mask.astype(bool)) >= thin_threshold else "blob"


def measure_thin(mask: np.ndarray, mm_per_px: float | None = None) -> GeometryMetrics:
    """크랙류: 스켈레톤 중심축 폭 프로파일 + 길이.

    - 폭 프로파일 = 2 × EDT[스켈레톤 픽셀]
    - 길이 = 스켈레톤 픽셀 수 × 1.05 (대각 연결 보정 근사)
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image 필요: pip install scikit-image")
    m = mask.astype(bool)
    s = mm_per_px if mm_per_px else 1.0
    unit = "mm" if mm_per_px else "px"

    edt = ndimage.distance_transform_edt(m)
    skel = skeletonize(m)
    widths_px = 2.0 * edt[skel]
    if widths_px.size == 0:
        return GeometryMetrics(kind="thin", unit=unit)

    cy, cx = ndimage.center_of_mass(m)
    return GeometryMetrics(
        kind="thin", unit=unit,
        area=float(m.sum()) * s * s,
        length=float(skel.sum()) * 1.05 * s,
        width_mean=float(widths_px.mean()) * s,
        width_max=float(widths_px.max()) * s,
        width_p95=float(np.percentile(widths_px, 95)) * s,
        width_profile=widths_px * s,
        centroid=(float(cy), float(cx)),
    )


def measure_blob(mask: np.ndarray, mm_per_px: float | None = None) -> GeometryMetrics:
    """부품류: 등가직경 + 주축/부축 (관성 모멘트 기반, cv2 불필요)."""
    m = mask.astype(bool)
    s = mm_per_px if mm_per_px else 1.0
    unit = "mm" if mm_per_px else "px"
    area_px = float(m.sum())
    if area_px == 0:
        return GeometryMetrics(kind="blob", unit=unit)

    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    # 공분산 고유값 → 주축/부축 길이 (가우시안 등가 4σ 근사)
    cov = np.cov(np.stack([ys - cy, xs - cx]))
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 4.0 * np.sqrt(max(evals[0], 0))
    minor = 4.0 * np.sqrt(max(evals[-1], 0))
    return GeometryMetrics(
        kind="blob", unit=unit,
        area=area_px * s * s,
        length=float(major) * s,
        width_mean=float(minor) * s,
        width_max=float(minor) * s,
        width_p95=float(minor) * s,
        equiv_diameter=float(2.0 * np.sqrt(area_px / np.pi)) * s,
        centroid=(float(cy), float(cx)),
    )


def measure(mask: np.ndarray, mm_per_px: float | None = None, kind: str = "auto") -> GeometryMetrics:
    """마스크 → 계측. kind="auto"면 형상으로 thin/blob 판별."""
    if kind == "auto":
        kind = classify_kind(mask)
    return measure_thin(mask, mm_per_px) if kind == "thin" else measure_blob(mask, mm_per_px)


def pairwise_spacing(metrics: list[GeometryMetrics], mm_per_px: float | None = None) -> np.ndarray:
    """인스턴스 간 중심 거리 행렬 (볼트 간격 등). centroid는 px 기준이므로 여기서 스케일."""
    s = mm_per_px if mm_per_px else 1.0
    c = np.array([m.centroid for m in metrics])
    if len(c) < 2:
        return np.empty((0, 0))
    return np.linalg.norm(c[:, None] - c[None], axis=2) * s
