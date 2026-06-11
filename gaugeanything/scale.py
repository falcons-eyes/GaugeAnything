"""스케일 리졸버 — 픽셀→mm 변환의 3가지 경로.

검증된 갭(VISION_DESIGN §3.1): 어떤 파운데이션 모델도 픽셀→mm를 못 한다.
우선순위 순 폴백:
  1. ArUco/ChArUco 마커 (cv2.aruco) — 현장 촬영 프로토콜의 기준
  2. 기지 치수 객체 (볼트머리 규격 M8=13mm 등) — 사용자/탐지기가 bbox 제공
  3. 수동 mm_per_px

평면 가정: 마커와 측정 대상이 같은 평면(벽면 크랙 등)일 때 유효.
깊이 차이가 있으면 DA-V2 상대깊이로 보정하는 v1 과제 (TODO).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
    _HAS_CV2 = hasattr(cv2, "aruco")
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# 표준 볼트머리 평면폭(across-flats, mm) — 기지 치수 레퍼런스
BOLT_HEAD_AF_MM = {"M4": 7.0, "M5": 8.0, "M6": 10.0, "M8": 13.0,
                   "M10": 16.0, "M12": 18.0, "M16": 24.0, "M20": 30.0}


@dataclass
class ScaleResult:
    mm_per_px: float
    method: str          # "aruco" | "known_object" | "manual"
    n_refs: int = 1      # 사용된 레퍼런스 수 (마커 개수 등)
    std: float = 0.0     # 레퍼런스 간 편차 (신뢰도 지표)


def from_manual(mm_per_px: float) -> ScaleResult:
    return ScaleResult(mm_per_px=mm_per_px, method="manual")


def from_known_object(bbox_px: tuple[float, float, float, float], real_size_mm: float,
                      axis: str = "long") -> ScaleResult:
    """기지 치수 객체의 bbox(x1,y1,x2,y2)와 실제 크기 → mm/px.
    axis: 'long'=장변 기준(기본), 'short'=단변, 'width'/'height'=축 지정."""
    w = abs(bbox_px[2] - bbox_px[0])
    h = abs(bbox_px[3] - bbox_px[1])
    px = {"long": max(w, h), "short": min(w, h), "width": w, "height": h}[axis]
    if px <= 0:
        raise ValueError("bbox 크기가 0")
    return ScaleResult(mm_per_px=real_size_mm / px, method="known_object")


def from_bolt_head(bbox_px: tuple[float, float, float, float], size: str = "M8") -> ScaleResult:
    """볼트머리 규격을 무료 스케일 레퍼런스로 사용 (현장에서 가장 흔한 기지 치수)."""
    r = from_known_object(bbox_px, BOLT_HEAD_AF_MM[size], axis="short")
    r.method = f"known_object:bolt_{size}"
    return r


def from_aruco(image: np.ndarray, marker_size_mm: float,
               dictionary: int | None = None) -> ScaleResult | None:
    """이미지에서 ArUco 마커 탐지 → 변 길이 픽셀 평균으로 mm/px.
    여러 마커가 보이면 평균 + 편차 보고. 탐지 실패 시 None."""
    if not _HAS_CV2:
        raise RuntimeError("opencv-contrib-python 필요 (cv2.aruco)")
    if dictionary is None:
        dictionary = cv2.aruco.DICT_4X4_50
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    side_px = []
    for c in corners:  # c: [1,4,2] 코너 좌표
        pts = c[0]
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        side_px.append(float(np.mean(sides)))
    ratios = [marker_size_mm / s for s in side_px]
    return ScaleResult(mm_per_px=float(np.mean(ratios)), method="aruco",
                       n_refs=len(side_px), std=float(np.std(ratios)))


def make_aruco_board(marker_size_px: int = 200, margin: int = 40,
                     dictionary: int | None = None, marker_id: int = 0) -> np.ndarray:
    """검증/인쇄용 ArUco 마커 이미지 생성 (selftest에서 GT로 사용)."""
    if not _HAS_CV2:
        raise RuntimeError("opencv-contrib-python 필요")
    if dictionary is None:
        dictionary = cv2.aruco.DICT_4X4_50
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    img = cv2.aruco.generateImageMarker(d, marker_id, marker_size_px)
    return cv2.copyMakeBorder(img, margin, margin, margin, margin,
                              cv2.BORDER_CONSTANT, value=255)


@dataclass
class PlaneScale:
    """Homography 기반 평면 스케일 — 틸트(perspective)에서도 정확한 mm 측정.

    스칼라 mm/px(정면 가정)와 달리, 마커 4코너 → mm 평면 homography H를 추정해
    픽셀 좌표를 마커 평면의 mm 좌표로 사상한다. 마커와 같은 평면 위 측정에 유효.
    """

    H: np.ndarray            # 3×3, px → mm (마커 평면 좌표)
    method: str = "aruco_homography"
    n_refs: int = 1

    def to_plane_mm(self, pts_px: np.ndarray) -> np.ndarray:
        """[N,2] 픽셀 → [N,2] mm 평면 좌표."""
        p = np.concatenate([pts_px.astype(np.float64), np.ones((len(pts_px), 1))], 1)
        q = (self.H @ p.T).T
        return q[:, :2] / q[:, 2:3]

    def distance_mm(self, p1, p2) -> float:
        a, b = self.to_plane_mm(np.array([p1, p2], np.float64))
        return float(np.linalg.norm(a - b))

    def local_mm_per_px(self, at_px) -> float:
        """해당 픽셀 근방의 국소 스케일 (1px 변위의 mm 크기 평균)."""
        x, y = at_px
        pts = np.array([[x, y], [x + 1, y], [x, y + 1]], np.float64)
        m = self.to_plane_mm(pts)
        return float((np.linalg.norm(m[1] - m[0]) + np.linalg.norm(m[2] - m[0])) / 2)


def plane_from_aruco(image: np.ndarray, marker_size_mm: float,
                     dictionary: int | None = None) -> PlaneScale | None:
    """ArUco 마커 코너 4점 → px→mm homography. 틸트 보정형 스케일 (from_aruco의 상위호환)."""
    if not _HAS_CV2:
        raise RuntimeError("opencv-contrib-python 필요")
    if dictionary is None:
        dictionary = cv2.aruco.DICT_4X4_50
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    src = corners[0][0].astype(np.float64)                      # 검출 코너 (px)
    s = marker_size_mm
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float64)  # mm 평면
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return None
    return PlaneScale(H=H, n_refs=len(corners))


def resolve(image: np.ndarray | None = None, *, marker_size_mm: float | None = None,
            ref_bbox: tuple | None = None, ref_size_mm: float | None = None,
            manual_mm_per_px: float | None = None) -> ScaleResult | None:
    """우선순위 폴백: aruco → known_object → manual. 전부 실패 시 None (px 단위 출력)."""
    if image is not None and marker_size_mm and _HAS_CV2:
        r = from_aruco(image, marker_size_mm)
        if r is not None:
            return r
    if ref_bbox is not None and ref_size_mm:
        return from_known_object(ref_bbox, ref_size_mm)
    if manual_mm_per_px:
        return from_manual(manual_mm_per_px)
    return None
