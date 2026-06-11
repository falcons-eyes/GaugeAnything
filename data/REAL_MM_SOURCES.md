# Real-Metric GT Substitutes — 공개 데이터 전수조사 (2026-06-11)

> 현장 촬영 없이 픽셀→mm 측정을 검증/학습할 수 있는 공개 데이터 지도.
> 3가지 메커니즘: ① 명시적 mm 라벨 ② CAD(mm)+카메라 포즈 → 임의 치수 유도 ③ 규격 치수 물체.
> 전 항목 라이선스·다운로드 경로 검증(2026-06-11). Kaggle 토큰 불필요 경로 우선.

## Tier S — 즉시 사용 (라이선스 클린 + 직접 다운로드)

| 데이터셋 | mm GT 메커니즘 | 규모 | 라이선스 | 경로 |
|---|---|---|---|---|
| **T-LESS** (BOP) | **산업부품 CAD(mm)+intrinsics+pose** → 보이는 모든 치수 유도 가능 | 30 부품, 48k+ | **CC BY 4.0** | HF `bop-benchmark/tless` (비gated, wget) |
| **HB** (BOP) | 동일 (33 객체, 산업 포함) | 13 scenes RGB-D | **CC0** | HF `bop-benchmark/hb` |
| **krkCMd** 콘크리트 크랙 | **명시적 크랙 폭 GT** 19,098개 (0–800μm) + 3.97μm/px 문서화 | 36–39GB | **CC BY 4.0** | Zenodo 14568863 |
| **SmartDoc15-CH1** | A4(210×297mm) **GT quad** ×~25k 프레임, 실제 perspective | 24k+ | 공개 | Zenodo 1230218 / GitHub jchazalon |
| **DeepFish(tray)** | tray homography 기반 **개체별 mm 길이** (전문가 검증 ~2-3%) | 1,320장 | **CC BY 4.0** | Zenodo |
| **kaa/coins-dataset** | 유로 동전(법정 직경: €1=23.25, €2=25.75mm) **top-down on A4** — 이중 스케일 참조 | 소규모+원본scene | **MIT** | GitHub clone |
| **Francesco/coins-1apki** | 동전 COCO 박스 (원형→박스변≈직경) | 8,419장 | CC계열(확인) | HF 비gated |
| **ROI-1555 Rebar** | — (카운팅: 박스+**마스크**) | 1,555장 | 미명시 | HF 비gated |
| **TimberSeg 1.0** | — (카운팅: 통나무 마스크 ~2.5k) | 220장 | **CC BY 4.0** | Mendeley 직접 |

## Tier A — 유용 (제약 있음)

| 데이터셋 | 메커니즘 | 제약 |
|---|---|---|
| MVTec ITODD (BOP) | 진짜 공장부품 28종 CAD+pose — 가장 산업적 | **CC BY-NC-SA** → 벤치마크 전용 |
| YCB-V (BOP) | 가정용품 CAD(MIT) | 산업성 낮음 |
| MVTec Screws | 13종 나사 **oriented box** (종→치수 매핑 가능) | NC + 폼 다운로드 |
| RebarDSC | 카운팅 2,125장/350k 박스 (최대 밀집) | 라이선스 미명시, GDrive |
| FSC-147 산업 서브셋 | screws/nails/nuts/bricks/logs/coins/legos… **dot GT** | 이미지 라이선스 불명(웹수집) → 평가 전용 |
| MIDV-500 | ID-1 카드 85.60×53.98mm quad GT | FTP 배포 |
| DTU MVS / Middlebury 2014 | sub-mm 깊이 GT (근거리) | 연구 관행 라이선스 |
| AutoFish | 어류 길이 cm GT 1,500장 | HF, 라이선스 미명시 |
| LEGO (Boiński Sci Data) | 부품ID→정확 치수(stud 8.0mm), 155k 실사 | 호스트가 봇 차단(브라우저 필요) |
| 회전 M20 볼트 (Data in Brief) | **M20 단일 규격** 1,100장 | 도메인 협소 |

## 사용 불가 판정 (함정 — 기록)
- DeepCrack/CFD/Crack500/SDNET 등 크랙셋: **픽셀 마스크만, 물리 단위 없음**
- DFUC 상처: 룰러가 보여도 **라벨에서 제외** + gated
- NPU-BOLT: 토큰-프리 경로 전멸 확인(서명링크 404, Dropbox 대역폭 캡, Kaggle/Roboflow는 계정) + 라이선스 미명시 → **rebar로 대체**
- Roboflow Universe 전반: "공개"여도 계정+API키 필수

## 검증 계획 (현장 대체 실험 설계)

1. **E-mm-1 동전/A4** (즉시): kaa 원본 scene → A4 quad 검출 → PlaneScale homography →
   SAM3 "coin" → 등가직경(mm) vs 법정 직경. **우리 파이프라인 그대로의 e2e 실측 검증.**
2. **E-mm-2 T-LESS** : CAD+pose로 부품의 보이는 치수 GT 유도 → inspect 측정과 비교
   (산업 부품에서의 mm 정확도).
3. **E-mm-3 krkCMd** : 명시적 크랙 폭 GT(μm)로 폭 측정 충실도 — **헤드라인 주장(크랙 폭)의
   물리 단위 검증.** 도메인은 스캐너 스택임을 명시.
4. **E-cnt-1 rebar** : ROI-1555(+RebarDSC)로 SAM3 카운팅 MAE — 로드맵 2단계 대체.

다운로드: `bash data/scripts/download_metric.sh`
