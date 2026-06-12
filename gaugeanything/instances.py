"""멀티-인스턴스 thin-구조 계측 — "mask=WHERE, signal=WIDTH"의 정식 API.

E-loop 사다리의 결론(2026-06-12): 장면에는 크랙이 복수일 수 있고, 계측기의 올바른
출력은 단일 선택이 아니라 **모든 인스턴스 각각의 측정**이다 (krkCMd 검증:
recall 93%, 폭 MAE 30.5μm/중앙 16.4μm — `experiments/krkcmd_multiinstance_eval.py`).

파이프라인:
  타일 SAM3("crack") → 연결 성분(인스턴스) → 성분별 스켈레톤 경로
  → 경로 위 측정 스테이션마다 수직 원신호 프로파일 (snap-to-valley)
  → 폭 추정기(기본: minrun 규칙 / 권장: profile CNN 체크포인트) → 인스턴스별 폭 통계

검증 범위(정직): 폭 추정기의 절대 μm 정확도는 6400dpi 스캐너 도메인(krkCMd)에서
검증됨. 다른 해상도/재질에서는 스케일 정규화 후 사용하고 게이트·σ와 함께 보고할 것.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SNAP = 80          # snap-to-valley 반경 (px)
N_PROFILE = 501    # 프로파일 길이 (krkCMd 규약과 동일)


@dataclass
class ThinInstance:
    """크랙 등 thin 구조 1개의 인스턴스 측정 결과."""
    path: dict          # {col: row} 스켈레톤 경로
    stations: list      # [(col, row)] 측정 지점
    widths_px: list     # 지점별 폭 (px 단위 추정기 출력)
    meta: dict = field(default_factory=dict)

    def width_stats(self, scale: float = 1.0) -> dict:
        """scale: 출력 단위 변환 계수 (예: μm/px 또는 mm/px)."""
        w = np.asarray(self.widths_px, float) * scale
        if not len(w):
            return {"n": 0}
        return {"n": int(len(w)), "mean": float(w.mean()),
                "median": float(np.median(w)), "p95": float(np.percentile(w, 95)),
                "max": float(w.max())}


def width_minrun(profile: np.ndarray, accuracy: float = 5.0, px: float = 1.0) -> float:
    """valley 국소 연속 run 폭 (결정적 기본 추정기, px 단위)."""
    c = int(np.argmin(profile))
    mask = profile <= float(profile[c] + accuracy)
    left = right = c
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < mask.size and mask[right + 1]:
        right += 1
    return (right - left + 1) * px


def _tile_union_mask(gray: np.ndarray, prompt: str, ds: int, tile: int,
                     overlap: float, threshold: float) -> np.ndarray:
    from .segmenters import segment_sam3
    g8 = np.clip(gray / max(gray.max(), 1) * 255, 0, 255).astype(np.uint8)
    small = g8[::ds, ::ds]
    h, w = small.shape
    m = np.zeros(small.shape, bool)
    stride = max(1, int(tile * (1 - overlap)))
    xs = sorted({min(x, max(w - tile, 0)) for x in range(0, max(w - tile, 0) + stride, stride)})
    ys = sorted({min(y, max(h - tile, 0)) for y in range(0, max(h - tile, 0) + stride, stride)})
    for y0 in ys:
        for x0 in xs:
            crop = small[y0:y0 + tile, x0:x0 + tile]
            for inst in segment_sam3(np.stack([crop] * 3, -1), prompt, threshold=threshold):
                m[y0:y0 + tile, x0:x0 + tile] |= inst.mask
    return m


def measure_thin_instances(gray: np.ndarray, prompt: str = "crack", *,
                           width_fn: Callable[[np.ndarray], float] = width_minrun,
                           ds: int = 2, tile: int = 1024, overlap: float = 0.2,
                           threshold: float = 0.3, min_span_frac: float = 0.05,
                           station_step: int = 100,
                           profile_len: int = N_PROFILE) -> list[ThinInstance]:
    """그레이 이미지에서 모든 thin 인스턴스를 찾아 각각 신호 기반 폭 측정.

    width_fn: 501px 프로파일 → 폭(px). 기본 minrun 규칙; profile CNN 사용 시
    `experiments/krkcmd_signal_width.build_1d_net` 체크포인트를 감싸 전달.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    H, W = gray.shape
    m = _tile_union_mask(gray, prompt, ds, tile, overlap, threshold)
    lab, n = ndimage.label(m)
    out: list[ThinInstance] = []
    for k in range(1, n + 1):
        comp = lab == k
        xs_k = np.nonzero(comp.any(0))[0]
        if not len(xs_k) or (xs_k[-1] - xs_k[0]) < min_span_frac * m.shape[1]:
            continue
        sk = skeletonize(comp)
        yy, xx = np.nonzero(sk)
        path: dict[int, list[int]] = {}
        for y, x in zip(yy, xx):
            path.setdefault(int(x) * ds, []).append(int(y) * ds)
        path_med = {c: int(np.median(v)) for c, v in path.items()}
        cols = sorted(path_med)
        stations, widths = [], []
        for c in cols[:: max(1, station_step // max(ds, 1))]:
            yh = path_med[c]
            lo = max(0, yh - SNAP); hi = min(H, yh + SNAP + 1)
            yc = lo + int(np.argmin(gray[lo:hi, c]))
            a = max(0, yc - profile_len // 2)
            p = gray[a: a + profile_len, c]
            if len(p) != profile_len:
                continue
            stations.append((c, yc))
            widths.append(float(width_fn(p)))
        if stations:
            out.append(ThinInstance(path=path_med, stations=stations,
                                    widths_px=widths,
                                    meta={"prompt": prompt, "n_path_cols": len(cols)}))
    return out
