# Data Acquisition Guide — Industrial Anything

> Step 1 산출물. 2026년 6월 기준 다운로드 링크·포맷·라이선스 검증 완료.
> **상업 배포 라이선스가 핵심 결정 필드.** 학습 코퍼스를 `commercial` / `research` 두 트랙으로 분리한다.

## 0. 라이선스 트랙 — 처음부터 분리하라

상업 제품(FalconEyes) 가중치 공개를 염두에 두면, **상업 가능 데이터만으로 학습한 가중치**와 **연구 전용 가중치**를 분리해야 한다. HuggingFace 공개 시 라이선스 오염을 피하는 유일한 방법.

| 트랙 | 데이터셋 | 라이선스 |
|---|---|---|
| **✅ COMMERCIAL** (자유 배포) | KAIST, Ottawa(UORED-VAFCLS), N-CMAPSS, PHM2010 milling, IMS, VisA, InfraredSolarModules, MIMII 원본 | CC BY / CC0 / 공공 / MIT / CC BY-SA |
| **⚠️ RESEARCH-ONLY** (논문/벤치만) | Paderborn(KAT), MaFaulDa, MIMII-DUE/DG, ToyADMOS2, DCASE, MVTec AD/LOCO, Real-IAD, FEMTO, XJTU-SY | NC / 라이선스 미명시 |

> **CC BY-SA(MIMII 원본) 주의**: share-alike 의무 → 파생 데이터셋도 CC BY-SA로 공개해야 함. 가중치만 배포하면 통상 문제없으나, 데이터 재배포 시 copyleft 전파.

---

## 1. 멀티모달 (동기화 다중 센서) — 사전학습 핵심

### KAIST Rotating Machine ✅ CC BY 4.0 (보유)
- **모달**: 진동 4ch@25.6kHz + 전류 3상@25.6kHz + 온도 2ch@25.6kHz + 음향 1ch@51.2kHz
- **다운로드** (Mendeley `ztmf3m7h5x` v3, 인증 불필요, ~4.26GB):
  - base: `https://data.mendeley.com/public-files/datasets/ztmf3m7h5x/files`
  - `vibration.zip` (2.66GB) `…/ee98c5d9-1052-4448-84b2-ed57711b658d/file_downloaded`
  - `current,temp.zip` (1.55GB) `…/4fe7c7e8-9a77-4bef-b359-d762b6a3a044/file_downloaded`
  - `acoustic.zip` (47MB) `…/99f03ffd-04f9-4676-932a-4312284cbc18/file_downloaded`
- **포맷**: 진동/음향 `.mat`, 전류/온도 **`.tdms`** (`pip install npTDMS` 필요)
- **고장**: normal, 베어링 내륜/외륜, 축 정렬불량, 회전체 불평형 / 부하 0·2·4Nm

### Ottawa UORED-VAFCLS ✅ CC BY 4.0
- **모달**: 진동 + 음향 + 부하 + 속도 + 온도 2ch, 42kHz, 10s
- **다운로드** (Mendeley `y2px5tg92h` v2, 인증 불필요):
  - 전체 파일 열거: `curl -s "https://data.mendeley.com/public-api/datasets/y2px5tg92h/files?version=2" | jq '.[] | {name, url: .content_details.download_url}'`
  - 가속도 스펙트로그램(1.02GB): `…/files/30e2ed6d-14a7-4f33-a4cd-5d86c0b33bb6/file_downloaded`
  - 음향 스펙트로그램(777MB): `…/files/03683140-c10f-43a4-92b5-e36c2df1b072/file_downloaded`
- **포맷**: raw `.mat`/`.csv`/`.xlsx` (5컬럼: accel/acoustic/speed/load/temp) + 스펙트로그램 `.png`

### Paderborn / KAT ⚠️ CC BY-NC 4.0 (연구 전용)
- **모달**: 진동 64kHz + **3상 전류** 64kHz + 온도/속도/토크/부하 — *유일한 대형 진동+전류 공개셋*
- **다운로드**: KAT 인덱스(전체) `https://groups.uni-paderborn.de/kat/BearingDataCenter/` (개별 `.rar`, `unrar` 필요) / Zenodo 부분집합(32파일) `https://zenodo.org/records/15845309`
- ⚠️ Zenodo는 CC BY 태그지만 원본은 CC BY-NC → **NC로 취급, 상업 시 저자 허가**

### MaFaulDa (UFRJ) ⚠️ 라이선스 미명시 (연구 전용)
- **모달**: 진동 6축 + 마이크 + 타코 (8컬럼), 50kHz, 5s, 1951 시퀀스
- **다운로드**: 공식 `https://www02.smt.ufrj.br/~offshore/mfs/page_01.html` (**TLS 인증서 만료** → `wget --no-check-certificate`) / Kaggle 미러 `kaggle datasets download -d vuxuancu/mafaulda-full` (권장)
- **포맷**: 헤더 없는 8컬럼 `.csv`

---

## 2. Run-to-Failure / RUL

> **NASA S3 마이그레이션**: 모든 논문이 인용하는 `ti.arc.nasa.gov/...` 링크는 전부 죽음. `https://phm-datasets.s3.amazonaws.com/NASA/...`로 교체 (익명 wget 가능).

### N-CMAPSS ✅ CC0 (상업 최적)
- **다운로드**: `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`
- **포맷**: HDF5 (`N-CMAPSS_DS01.h5`~`DS08.h5`). 그룹: `W`(운전조건) `X_s`(14 센서) `X_v`(가상센서) `Y`(**RUL 라벨**)
- **주의**: 구 C-MAPSS(#6, FD001-004 txt)와 혼동 금지 — N-CMAPSS는 **#17, HDF5**

### IMS Bearing ✅ 공공(US Gov) (보유)
- **다운로드**: `https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip` (fallback `https://data.nasa.gov/docs/legacy/IMS.zip`)
- **포맷**: 확장자 없는 ASCII(탭구분). 9,463 파일, 20kHz, 1s=20,480pt, 10분 간격

### PHM 2010 Milling ✅ CC0 (Kaggle)
- **다운로드**: `kaggle datasets download -d rabahba/phm-data-challenge-2010`
- **모달**: 절삭력 3축 + 진동 3축 + AE-RMS = **7ch@50kHz**. c1/c4/c6만 마모 라벨
- **주의**: NASA "Milling"(#3)과 다른 데이터셋

### FEMTO / PRONOSTIA ⚠️ 라이선스 미명시 (연구)
- **다운로드**: `https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip` / GitHub 미러 `wkzs111/phm-ieee-2012-data-challenge-dataset`
- **모달**: 진동 2축@25.6kHz + 온도@10Hz, 17 베어링 run-to-failure

### XJTU-SY ⚠️ citation-only (연구)
- **다운로드**: 클라우드 드라이브만 → `gdown --folder "https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK"`
- **모달**: 진동 2축@25.6kHz, 15 베어링 run-to-failure

---

## 3. 음향 (이상탐지)

### MIMII 원본 ✅ CC BY-SA 4.0 (상업 OK, share-alike)
- **다운로드**: Zenodo `3384388` → `zenodo_get 3384388` (~100GB, 8ch 16kHz 10s)

### MIMII-DUE/DG ❌, ToyADMOS2 ❌, DCASE ❌ (전부 NC)
- MIMII-DUE `zenodo_get 4740355` / MIMII-DG `zenodo_get 6529888` (CC BY-NC-SA)
- ToyADMOS2 `zenodo_get 4580270` (NTT 커스텀, **재배포 금지 — 가장 엄격**)
- DCASE 2025 Task2 `zenodo_get 15097779 15519362` (+2024 `10902294 11259435`) — 16kHz mono 6-10s, CC BY-NC-SA

---

## 4. 비전 / 열화상 / 표면결함

### VisA ✅ CC BY 4.0 (비전 상업 최적)
- **다운로드**: `wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar && tar -xf VisA_20220922.tar`
- 10,821 이미지, 12 클래스. 툴링 `github.com/amazon-science/spot-diff`
- ⚠️ 일부 서드파티 문서가 NC로 오표기 — **공식 Amazon = CC BY 4.0**

### InfraredSolarModules ✅ MIT (열화상 상업 최적)
- **다운로드**: `git clone https://github.com/RaptorMaps/InfraredSolarModules.git` → `2020-02-14_InfraredSolarModules.zip` 압축해제
- 20,000장 24×40 열화상 IR, 12 클래스

### MVTec AD/LOCO ❌, Real-IAD ❌ (전부 NC)
- MVTec AD: `https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads` (~4.9GB, CC BY-NC-SA)
- Real-IAD: HF `Real-IAD/Real-IAD` (gated 자동승인, `realiad_1024` ~53GB, CC BY-NC-SA)
- Real-IAD D³ (RGB+3D 멀티모달): **2026.6 기준 미공개** ("Coming soon") — 상업 파이프라인 설계 금지

---

## 5. 포맷 게처(gotcha) 요약

| 데이터셋 | 필요 도구 |
|---|---|
| KAIST 전류/온도 | `pip install npTDMS` (.tdms) |
| Paderborn | `unrar` / `7z` (.rar) |
| MaFaulDa | 헤더 없는 8컬럼 CSV; 공식 호스트 TLS 만료 → 미러 |
| N-CMAPSS | `h5py` (HDF5, 그룹 구조) |
| IMS | 확장자 없는 ASCII 탭구분 |
| XJTU-SY | `gdown` (드라이브) |
| 음향 NC셋 | `pip install zenodo_get` |

---

## 6. 다음 작업
- `scripts/download_commercial.sh` — 상업 트랙 자동 다운로드
- `scripts/download_research.sh` — 연구 트랙 (논문/벤치용)
- → Step 2: `src/harmonization/` 공통 타임라인 정렬 PoC (KAIST + Paderborn)
