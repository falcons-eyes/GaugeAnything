"""GaugeAnything v0 — 합성 파이프라인 (Stage 0).

이미지 + 프롬프트 → 세그멘터 → 인스턴스 마스크 → 스케일 리졸버 → 기하 계측
→ Inspection Atom 집합 {mask, label, count, metrics(mm±σ), confidence}

VISION_DESIGN §4의 Promptable Quantitative Inspection 태스크 정식화 구현.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .geometry import GeometryMetrics, measure, pairwise_spacing
from .scale import ScaleResult, resolve
from .segmenters import Instance, get_segmenter


@dataclass
class InspectionAtom:
    """원자 검사 단위 — Parallel Inspection Decoding의 출력 단위 (VISION_DESIGN §4)."""

    label: str
    confidence: float
    metrics: GeometryMetrics
    instance_id: int = 0

    def to_dict(self) -> dict:
        return {"id": self.instance_id, "label": self.label,
                "confidence": round(self.confidence, 3), **self.metrics.scaled()}


@dataclass
class InspectionResult:
    atoms: list[InspectionAtom] = field(default_factory=list)
    count: int = 0
    scale: ScaleResult | None = None
    spacing_mm: np.ndarray | None = None  # 인스턴스 간 거리 행렬

    def summary(self) -> str:
        lines = [f"인스턴스: {self.count}개"]
        if self.scale:
            lines.append(f"스케일: {self.scale.mm_per_px:.4f} mm/px "
                         f"({self.scale.method}, refs={self.scale.n_refs}, σ={self.scale.std:.4f})")
        else:
            lines.append("스케일: 미해석 → px 단위")
        for a in self.atoms:
            m = a.metrics
            if m.kind == "thin":
                lines.append(f"  [{a.instance_id}] {a.label}: 폭 평균 {m.width_mean:.2f}{m.unit} "
                             f"/ 최대 {m.width_max:.2f} / p95 {m.width_p95:.2f}, "
                             f"길이 {m.length:.1f}{m.unit} (conf {a.confidence:.2f})")
            else:
                lines.append(f"  [{a.instance_id}] {a.label}: 등가직경 {m.equiv_diameter:.2f}{m.unit}, "
                             f"{m.length:.1f}×{m.width_mean:.1f}{m.unit} (conf {a.confidence:.2f})")
        if self.spacing_mm is not None and self.spacing_mm.size:
            iu = np.triu_indices_from(self.spacing_mm, k=1)
            vals = self.spacing_mm[iu]
            lines.append(f"  간격: 평균 {vals.mean():.1f} / 최소 {vals.min():.1f}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "count": self.count,
            "scale": ({"mm_per_px": self.scale.mm_per_px, "method": self.scale.method}
                      if self.scale else None),
            "atoms": [a.to_dict() for a in self.atoms],
        }, ensure_ascii=False, indent=2)


def inspect(
    image: np.ndarray,
    prompt: str,
    *,
    segmenter: str = "sam3",
    kind: str = "auto",
    # 스케일 입력 (우선순위 폴백: aruco → known_object → manual)
    marker_size_mm: float | None = None,
    ref_bbox: tuple | None = None,
    ref_size_mm: float | None = None,
    manual_mm_per_px: float | None = None,
    max_instances: int = 100,
) -> InspectionResult:
    """GaugeAnything v0 진입점."""
    seg = get_segmenter(segmenter)
    instances: list[Instance] = seg(image, prompt)[:max_instances]

    scale = resolve(image, marker_size_mm=marker_size_mm, ref_bbox=ref_bbox,
                    ref_size_mm=ref_size_mm, manual_mm_per_px=manual_mm_per_px)
    mm_per_px = scale.mm_per_px if scale else None

    atoms = []
    for i, inst in enumerate(instances):
        g = measure(inst.mask, mm_per_px=mm_per_px, kind=kind)
        atoms.append(InspectionAtom(label=inst.label, confidence=inst.score,
                                    metrics=g, instance_id=i))
    spacing = pairwise_spacing([a.metrics for a in atoms], mm_per_px) if len(atoms) > 1 else None
    return InspectionResult(atoms=atoms, count=len(atoms), scale=scale, spacing_mm=spacing)
