"""Regime 라우터 — 결함 경계 성질로 binary/matting/field 자동 분기 (SOFT_INSPECTION.md §5).

결정 흐름:
  SAM3 마스크 약함/없음           → "field"  : 조명잔차 soft map (uneven/mura)
  마스크 있음 & 경계 흐림(저그래디언트) → "fuzzy" : guided_matte → soft α (fray)
  마스크 있음 & 경계 선명         → "sharp"  : binary 마스크 (크랙/홀)
그 다음 soft 측정 모듈로 통일 (Σα 면적·sub-pixel 폭·severity·±CI).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .geometry import classify_kind
from .segmenters import get_segmenter
from .soft import (
    area_uncertainty,
    guided_matte,
    mura_severity,
    severity_score,
    soft_area,
    soft_iso_length,
    soft_width,
)


def _gray(image):
    return (image if image.ndim == 2 else image.mean(2)).astype(np.float32)


def boundary_sharpness(gray: np.ndarray, mask: np.ndarray, band: int = 2) -> float:
    """마스크 경계 밴드의 평균 이미지 그래디언트 (선명할수록 큼)."""
    b = ndimage.binary_dilation(mask, iterations=band) ^ ndimage.binary_erosion(mask, iterations=band)
    if b.sum() < 10:
        return 0.0
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad[b].mean())


def classify_regime(image: np.ndarray, mask: np.ndarray, score: float,
                    min_px: int = 50, sharp_ratio: float = 0.55) -> str:
    """'field' | 'fuzzy' | 'sharp'. sharp_ratio: 경계/전역 그래디언트 비 임계."""
    gray = _gray(image)
    if mask.sum() < min_px or score < 0.3:
        return "field"
    gy, gx = np.gradient(gray)
    ref = float(np.percentile(np.sqrt(gx ** 2 + gy ** 2), 90)) + 1e-6
    return "sharp" if boundary_sharpness(gray, mask) >= sharp_ratio * ref else "fuzzy"


@dataclass
class SoftAtom:
    regime: str                      # sharp | fuzzy | field
    label: str
    confidence: float
    alpha: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        if self.regime == "field":
            return (f"[field] {self.label}: severity {m.get('grade','?')} "
                    f"(Sa={m.get('Sa',0):.2f}, Sq={m.get('Sq',0):.2f}, extent={m.get('extent',0):.3f})")
        meas = (f"width {m.get('width_mean',0):.2f}{m.get('unit','px')}"
                if m.get("kind") == "thin" else f"area {m.get('area',0):.0f}{m.get('unit','px')}²")
        return (f"[{self.regime}] {self.label}: {meas} ± {m.get('area_std',0):.1f}, "
                f"conf {self.confidence:.2f}")


def inspect_soft(image: np.ndarray, prompt: str, *, segmenter: str = "sam3",
                 mm_per_px: float | None = None, marker_size_mm: float | None = None,
                 max_instances: int = 50) -> list[SoftAtom]:
    """regime 라우팅 + soft 측정. SAM3 실패 시 field로 폴백.

    - segmenter="sam3_ensemble": 프롬프트 동의어 붕괴 방지 (PROMPT_SETS 매핑)
    - marker_size_mm: ArUco 검출 시 PlaneScale(homography)로 틸트 보정 mm —
      인스턴스 중심 위치의 국소 스케일 사용 (정면 가정 탈피, 50° 틸트에서도 ~1% 오차)
    """
    seg = get_segmenter(segmenter)
    insts = seg(image, prompt)[:max_instances]
    gray = _gray(image)

    plane = None
    if marker_size_mm:
        from .scale import plane_from_aruco
        try:
            plane = plane_from_aruco(image, marker_size_mm)
        except RuntimeError:
            plane = None
    unit = "mm" if (mm_per_px or plane) else "px"

    # 전역 field 폴백: 의미있는 인스턴스가 없으면 mura field 1개 반환
    strong = [i for i in insts if i.mask.sum() >= 50 and i.score >= 0.3]
    if not strong:
        sev = mura_severity(image)
        soft = sev["soft_map"]
        sc = severity_score(soft)
        return [SoftAtom(regime="field", label=prompt, confidence=0.0, alpha=soft,
                         metrics={**{k: sev[k] for k in ("Sa", "Sq", "Ssk", "Sku")}, **sc})]

    atoms = []
    for inst in strong:
        reg = classify_regime(image, inst.mask, inst.score)
        if reg == "fuzzy":
            alpha = guided_matte(image, inst.mask)        # soft α (경계 보존)
        else:                                             # sharp → binary를 α로
            alpha = inst.mask.astype(np.float32)
        # 인스턴스별 스케일: PlaneScale 우선(틸트 보정, 국소), 폴백 수동 스칼라
        scale = mm_per_px
        if plane is not None:
            cy, cx = ndimage.center_of_mass(inst.mask)
            scale = plane.local_mm_per_px((cx, cy))
        kind = classify_kind(alpha >= 0.5)
        m = {"kind": kind, "unit": unit,
             "area": soft_area(alpha, scale),
             "area_std": area_uncertainty(alpha, scale)}
        if kind == "thin":
            m.update(soft_width(alpha, mm_per_px=scale))          # width_mean, p95
            m["iso_length"] = soft_iso_length(alpha, mm_per_px=scale)
        else:
            m["equiv_diameter"] = float(2 * np.sqrt(m["area"] / np.pi)) if m["area"] > 0 else 0.0
        atoms.append(SoftAtom(regime=reg, label=inst.label or prompt,
                              confidence=inst.score, alpha=alpha, metrics=m))
    return atoms
