# Datasets — Paper Reference (GaugeAnything)

> 논문 Datasets/Experimental Setup 절을 위한 전수 정리. "역할"은 본 연구에서의 실제/계획
> 용도. 라이선스는 2026-06 검증 시점 기준 (인용 전 원문 재확인).
> 모든 사용 데이터의 다운로드 스크립트: `data/scripts/`.

## 1. 본 연구에서 사용함 (실험 수행 완료)

| 데이터셋 | 역할 | 내용/주석 | 규모(사용분) | 라이선스 | 출처 |
|---|---|---|---|---|---|
| **CrackSeg9k** | 분할·측정 벤치 (Gauge-Bench), M2 refiner 학습/홀드아웃 | 크랙 분할 마스크, 14개 원본 소스 통합(파일 접두사로 식별) | 9,159쌍 | CC0 (DeepCrack/GAPs 서브셋은 원본 NC — 상업 학습 시 제외 규칙) | Harvard Dataverse DVN/EGIEBY; Kulkarni+ 2022 |
| **Magnetic-Tile Defect** | 멀티도메인 일반화, soft 검사(3-regime), 학습형 헤드(DRAEM/matting) 학습·평가 | 결함 5종+정상, 픽셀 GT 마스크 | 1,344장 (Free 952 학습풀) | 미명시(연구 관행) | Huang+ (github abin24); Surface-defect-detection 계열 |
| **VT LCW** (Bianchi&Hebdon) | 크랙 도메인 보조 (다운로드 완료, 평가 일부) | 실제 교량검사 크랙 분할 | 3,817장 | **CC0** | VT figshare 16624672 |
| 합성 (자체) | 계측 self-test GT, matting/DRAEM 학습 신호 | 폭 기지 크랙, ArUco 렌더, 방향성 fray(I=αF+(1−α)B), 저주파 mura | — | 자체 | 본 repo 코드 |

주: 모 프로젝트(Industrial Anything)의 진동/PHM 데이터(KAIST·Paderborn·N-CMAPSS 등)는
본 논문 범위 밖 — 평가 방법론(leakage/confound 교훈)만 승계.

## 2. 확보 완료, 실험 예정 (mm GT 대체 트랙)

| 데이터셋 | 계획 역할 | mm GT 메커니즘 | 규모 | 라이선스 | 출처 |
|---|---|---|---|---|---|
| **kaa euro coins (src scenes)** | E-mm-1: known-object 스케일 e2e 검증 | 권종별 폴더(법정 직경 16.25–25.75mm); ⚠️ 실측 확인: A4 아닌 테이블 위 → 동일권종 cross-coin 검증으로 설계 변경 | 2,872장 | **MIT** | github kaa/coins-dataset |
| **ROI-1555 Rebar** | E-cnt-1: 카운팅 MAE | 박스+인스턴스 마스크 | 1,555장 | 미명시 | HF tsrobcvai/ROI-1555 |
| **SmartDoc15-CH1** | PlaneScale 실사진 검증 | A4(210×297mm) GT quad, 실제 perspective | ~25k 프레임 | 공개 | Zenodo 1230218 |
| **T-LESS** (BOP) | E-mm-2: 산업부품 치수 유도 검증 | CAD(mm)+intrinsics+pose | 30부품 | **CC BY 4.0** | HF bop-benchmark/tless |
| **krkCMd** | E-mm-3: 크랙 폭 물리 GT 검증 | 명시적 폭 측정 19,098개(0–800μm), 3.97μm/px | 36GB | **CC BY 4.0** | Zenodo 14568863; Jakubowski&Tomczak, Sci Data 2025 |

## 3. 조사 완료, 후보 (필요 시)

| 데이터셋 | 잠재 역할 | 핵심 | 라이선스 | 비고 |
|---|---|---|---|---|
| HB / YCB-V / LM (BOP) | mm 유도 보조 | CAD+pose | CC0 / MIT / CC BY | HF 비gated |
| MVTec ITODD | 산업부품 mm 유도 | 진짜 공장부품 28종 | **NC** | 벤치 전용 |
| RebarDSC | 카운팅 대규모 | 2,125장/350k 박스 | 미명시 | GDrive |
| FSC-147 산업 서브셋 | 카운팅 평가 | dot GT (screws/nuts/bricks/logs/coins…) | 이미지 불명 | 평가 전용 |
| DeepFish(tray) / AutoFish | mm 회귀 보조 | 개체별 mm/cm 길이 | CC BY / 미명시 | 도메인 상이 |
| MIDV-500 | 카드(85.60×53.98mm) quad | 규격 치수 | 공개 | FTP |
| MVTec Screws | 파스너 oriented box | 13종→치수 매핑 가능 | **NC** | 폼 |
| Francesco/coins-1apki | 동전 박스 8.4k | 직경≈박스변 | CC계열 | HF |
| SDNET2018 / METU / DAGM | 분류 볼륨/텍스처 | — | CC BY | 보조 |
| DTU MVS / Middlebury 2014 | 근거리 깊이 GT | sub-mm | 연구 관행 | 보조 |
| LEGO (Boiński) | stud 8.0mm 규격 | 155k 실사 | 공개(봇차단) | 보조 |
| dacl10k / Sewer-ML / MVTec AD·LOCO | 도메인 확장 평가 | 분할/분류 | **NC** | 벤치 전용 |

## 4. 사용 불가 판정 (논문 한계/관련연구 절에 인용 가치)

| 대상 | 사유 |
|---|---|
| DeepCrack/CFD/Crack500/CrackTree/SDNET 등 "크랙 측정"류 | **픽셀 마스크만, 물리 단위 없음** — 분야 전반의 mm GT 공백을 입증하는 근거 |
| DFUC 상처 시리즈 | 룰러가 찍혀도 라벨에서 의도적 제외 + gated |
| NPU-BOLT | 토큰-프리 배포 경로 전멸(2026-06 실측) + 라이선스 미명시 |
| UAV+레이저 크랙 측정 논문들 | 방법만 공개, 캘리브레이션된 데이터 미공개 |
| Adobe Comp-1k 기반 matting 가중치 전반 | 비상업 데이터 오염 → 자체 합성 학습으로 우회 |

## 인용 체크리스트 (논문 작성 시)
- CrackSeg9k: Kulkarni et al., "CrackSeg9k", ECCV-W 2022 + Dataverse DOI
- Magnetic-Tile: Huang et al., "Surface defect saliency of magnetic tile", Vis Comput 2020
- krkCMd: Jakubowski & Tomczak, Scientific Data 2025, DOI 10.5281/zenodo.14568398
- T-LESS: Hodaň et al., WACV 2017 + BOP (Hodaň et al., ECCV-W)
- SmartDoc: Burie et al., ICDAR 2015 competition
- LCW: Bianchi & Hebdon, J Comput Civ Eng / figshare DOI
