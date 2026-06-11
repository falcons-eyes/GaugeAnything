# Rigor Audit — 전체 비전 트랙 재감사 (2026-06-11)

> "발표 전에 우리가 우리를 먼저 공격한다" 2차. 모든 실험·주장·코드의 엄밀성 결함,
> 모델링 결함, 사이드 이펙트를 빠짐없이 기록하고 수정 계획을 박는다.
> 등급: 🔴 결론을 바꿀 수 있음 / 🟡 약화시킴 / 🟢 사소함.

## A. 평가 방법론 결함

### A1. 🔴 DRAEM/soft 평가의 test-set 최적화
DRAEM v1(0.593) → 합성·앙상블 수정 → **같은 테스트셋**에서 v2(0.639) 보고. 고전 조명잔차의
detrend+smooth 개선(0.566→0.683)도 같은 셋에서 반복 튜닝. 전형적인 test-set optimization.
**수정**: MT_Uneven을 고정 시드로 val/test 분할 → 모든 설정 선택은 val, 보고는 test만.

### A2. 🔴 gauge_bench mIoU에 noncrack(빈 GT) 혼입 — SAM3 유리한 편향
`iou_f1`이 빈GT+빈예측=1.0을 반환. noncrack 소스(1,411쌍)가 샘플에 포함되어, "아무것도
탐지 안 함"이 가능한 SAM3는 1.0을 받고 항상 무언가를 분할하는 고전(adaptive 등)은 0을 받음.
per-source에서 SAM3 noncrack=0.625 vs adaptive=0.0 — **mIoU 0.450에 탐지 능력이 섞여
분할 능력이 과대 표현**. 2.6× 배율이 줄어들 수 있음.
**수정**: crack-only mIoU와 noncrack 거짓양성을 분리 보고.

### A3. 🟡 binary AUC 비교의 구조적 불공정
binary 마스크는 ROC 곡선이 한 점 → AUC가 구조적으로 ~0.5 부근. "binary=랜덤(0.499)"
프레이밍은 연속맵에 유리한 지표 선택. binary 실패의 진짜 증거는 IoU≤0.03이며 그걸로 충분.
**수정**: 페이지/RESULTS에 캐벗 명기, IoU를 1차 증거로.

### A4. 🟡 단일 시드·소규모 n·신뢰구간 부재 (비전 트랙 전체)
gauge_bench n≈100-205·시드1, multidomain 도메인당 30, soft eval 25, fray 전체 32장.
센서 트랙은 시드 3개였는데 비전 트랙은 전부 시드 1개.
**수정**: gauge_bench 시드≥3 + mean±std. 소표본 도메인은 "전수 평가"로 명시.

### A5. 🟡 프롬프트 민감도 미측정
도메인당 프롬프트 1개(fray만 2개 시도). SAM3 zero-shot 수치가 프롬프트 선택의 함수인데
민감도를 모름.
**수정**: 프롬프트 스윕(도메인당 4개) → mean/best 보고.

### A6. 🟡 supervised 상한 부재 — 포지셔닝 왜곡 위험
페이지가 SAM3 0.45를 "고전의 2.6×"로 제시하나, CrackSeg9k에서 지도학습 U-Net류는
mIoU ~0.7+ 가 통상. zero-shot promptable의 가치는 별개지만 상한을 숨기면 오해 유발.
**수정**: 문헌 상한을 명시("supervised reference ~0.7+, ours is zero-shot/promptable").

## B. 학습형 결과의 검증 범위 결함

### B1. 🔴 Matting 14.5×는 합성→합성 + 부분 순환
α-MAE 0.0138 vs 0.2007의 binary 기준은 **우리가 직접 열화시킨 coarse 마스크**이고,
평가도 학습과 같은 합성 분포의 holdout. 네트워크가 "우리가 만든 열화의 역변환"을 배운 것일
가능성. 실제 fray에서의 검증 전무.
**수정**: 실제 MT_Fray에서 (i) α≥0.5 vs binary GT IoU(보존성), (ii) 경계 softness 분포,
(iii) 정성 패널. α GT 부재의 한계는 명시 유지.

### B2. 🟡 합성 fray 분포 갭
synth_fray = 페더링된 블롭. 실제 fray는 방향성 텍스처 손상. 분포 갭 미측정.

### B3. 🔴 체크포인트 미저장 — 재현 불능
DRAEM·matting 모두 state_dict 저장 없음. 보고된 모델이 디스크에 존재하지 않음.
HF 모델 공개가 목표인 프로젝트에서 모순.
**수정**: --save 추가, `checkpoints/` 규약, 이후 모든 학습은 저장 의무.

## C. 모델링/코드 사이드 이펙트

### C1. 🟡 `mura_severity(detrend_cols=True)` 기본값 = 데이터셋 특화 핵
자성타일의 수직 연삭 텍스처용 컬럼 detrend가 **기본값** → 다른 도메인에서 세로 방향
실제 결함을 지워버릴 수 있음. **수정**: 기본 False, 호출부에서 명시.

### C2. 🟡 severity_score의 자의적 상수
log 전이·가중 0.5/0.5·등급 경계(×4) 전부 미보정 발명품. ASTM "스타일"이지 ASTM 보정이
아님. **수정**: "uncalibrated heuristic" 명시, 등급 GT 확보 전까지 점수만(등급 라벨 보류).

### C3. 🟡 soft_width 분기 스켈레톤 편향
width=Σα/skeleton_length는 분기·다중 컴포넌트에서 평균 폭 왜곡(burr 문제의 soft판).
**수정 예정**: 수직 프로파일 적분(per-skeleton-point)으로 교체.

### C4. 🟢 SAM3 threshold 0.4/0.5 스크립트 간 불일치 — 통일 필요.
### C5. 🟢 라우터 임계(min_px=50, sharp_ratio=0.55) 수동 설정, 라우터 정확도 미정량
(uneven의 13/24가 sharp로 누수 확인). 결함 클래스→기대 regime 매핑으로 혼동행렬 측정 예정.
### C6. 🟢 ximgproc 가용성 — 검증 완료(✓ 4.13.0, guided filter 경로 실사용 확인).

## D. 주장-증거 갭 (페이지/문서)

### D1. 🟡 "1.53 mm end-to-end measured" — mm는 **가정 스케일**(0.25mm/px)
실측 mm GT 0건. ChArUco는 합성 검증만. 캡션엔 assumed 표기가 있으나 stat strip은 과장.
**수정**: "(assumed scale)" 명기. 실 mm 검증은 §F 참조.
### D2. 🟡 "직접 증명(directly proven)" 류 표현 — n=25 단일 시드엔 과한 단어. 완화.
### D3. 🟢 Architecture의 Count/SAHI — 실험 0건인 능력이 도식에 존재. "planned" 표기.

## E. 데이터 결함
- MT_Fray 32장·MT_Uneven 103장: 3-regime 주장의 실증이 **단일 데이터셋·단일 텍스처**에 의존.
- CrackSeg9k 내 NC 서브셋(DeepCrack/GAPs)이 평가 샘플에 포함 — 연구용은 적법하나 상업
  트랙 수치엔 분리 필요.
- 볼트/카운팅: 데이터 자체가 없음(능력 미실증).

## F. 최종 목표 대비 구조적 갭
1. **"GaugeAnything 모델"이 가중치로 존재하지 않음** — 현재는 합성 파이프라인 + 헤드 2개(미저장).
   HF 공개를 위해선 최소: matting 헤드 + field 헤드 가중치 + (M2) SAM3 fine-tune.
2. **실 mm GT 0건** — 모트라고 선언한 데이터가 아직 없음. 현장 촬영 전 대안:
   (i) 기하학적 perspective 시뮬레이션으로 스케일 리졸버 오차 정량(틸트 각도별),
   (ii) 물리 스케일이 정의된 고품질 합성 벤치(카메라 내참수 명시) — "synthetic-metric" 트랙으로 명명.
3. 카운팅·간격 미실증.

## 수정 실행 (이번 라운드)
- [x] C6 ximgproc 확인
- [ ] A2: gauge_bench crack-only/noncrack 분리 + A4 시드 3개 → 재실행
- [ ] A1: uneven val/test 프로토콜 (고전 설정 선택 + DRAEM 재학습·저장) → test만 보고
- [ ] B1+B3: matting 재학습(저장) + 실 MT_Fray 검증 패널/지표
- [ ] A5: 프롬프트 스윕 (crack/blowhole 도메인)
- [ ] C1: detrend_cols 기본 False
- [ ] D1/D2/A3/A6: 페이지·RESULTS 문구 정정
