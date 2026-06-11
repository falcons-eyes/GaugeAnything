"""GaugeAnything — promptable 정량 검사 (Stage 0 합성 파이프라인).

inspect(image, prompt, ...) -> InspectionResult {atoms, count, scale, spacing}
"""
from .geometry import GeometryMetrics, measure, pairwise_spacing
from .pipeline import InspectionAtom, InspectionResult, inspect
from .router import SoftAtom, classify_regime, inspect_soft
from .scale import BOLT_HEAD_AF_MM, ScaleResult, from_aruco, from_bolt_head, resolve
from .segmenters import SEGMENTERS, Instance, get_segmenter
from .soft import guided_matte, illumination_residual, mura_severity, severity_score

__all__ = [
    "inspect", "InspectionResult", "InspectionAtom",
    "inspect_soft", "SoftAtom", "classify_regime",
    "guided_matte", "illumination_residual", "mura_severity", "severity_score",
    "measure", "GeometryMetrics", "pairwise_spacing",
    "resolve", "ScaleResult", "from_aruco", "from_bolt_head", "BOLT_HEAD_AF_MM",
    "get_segmenter", "SEGMENTERS", "Instance",
]
