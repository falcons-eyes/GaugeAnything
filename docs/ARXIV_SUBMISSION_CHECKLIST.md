# arXiv 제출 체크리스트 — GaugeAnything paper v2

상태: PDF는 빌드·배포 완료([project page](https://falcons-eyes.github.io/GaugeAnything/static/pdfs/gaugeanything.pdf)).
남은 차단 요소는 **엔도스먼트**다 — 이것만 사용자(Hyunwoo Joo) 액션이 필요하다.

## 사용자 액션이 필요한 항목 (Claude가 대신 못 함)

### 1. arXiv 엔도스먼트 확보 (최우선)

cs.CV에 첫 제출하려면 엔도스먼트가 필요할 수 있다. 옵션(빠른 순):

1. **자동 면제 확인**: 등록 기관 이메일(.edu/연구기관) 또는 과거 arXiv 활동이 있으면
   엔도스먼트 없이 제출 가능. arxiv.org 계정 생성 후 cs.CV 제출 화면에서 면제 여부 즉시 표시.
2. **엔도서 직접 요청**: cs.CV에 최근 논문이 있는 지인/공동연구자에게 요청. arXiv가
   제출 시 부여하는 endorsement code를 전달하면 됨. 인접 저자 후보:
   - krkCMd 저자 (Jakubowski & Tomczak) — 데이터 인용 관계, 협조 가능성
   - OmniCrack30k (Benz & Rodehorst) — crack 분할 커뮤니티
   - 국내 비전/토목 SHM 연구실 컨택
3. **워크숍/OpenReview 병행**: 엔도스먼트가 지연되면 CVPR/ICCV 워크숍이나
   OpenReview에 먼저 올려 타임스탬프 확보 (선점 헤지).

### 2. 저자/소속 메타데이터 최종 확인

- 현재: Hyunwoo Joo, Falcon Eyes Inc. (CITATION.cff·main.tex·HF 카드 일치 확인됨)
- arXiv 제출 폼의 author/affiliation/category(primary cs.CV, cross-list cs.LG/eess.IV)
  입력은 수동.

### 3. 라이선스 선택

- arXiv 제출 시 라이선스 선택 필요. 권장: **CC BY 4.0** (오픈 연구 노선과 일관).
  코드는 이미 Apache-2.0.

## Claude가 이미 처리한 제출 준비 (재확인용)

- [x] main.tex 단일 파일 + refs.bib, Overleaf/표준 article 호환 (pdflatex 빌드 검증)
- [x] 모든 figure가 `paper/figures/`에 포함 (gauge_demo, coins_mm, krkcmd_profile 등)
- [x] 인용 전부 해소 (undefined citation 0 — Spark 빌드 로그 확인)
- [x] 수치 ↔ canonical JSON 정합 + GPU 변동 주석 (2026-06-13 drift fix)
- [x] GitHub/HF/project page URL 본문 일치

## camera-ready 잔여 (제출 후/개정 시, 비차단)

RELATED_BASELINES.md의 재확인 목록 + 이번 세션 발견:

- multiinstance·signal-CNN 외 다른 e2e 수치도 canonical JSON 전수 대조 (이번엔 2건만 정합)
- CrackMamba를 우리 split으로 재실행 (코드 공개됨)
- 카운팅 표에 CountGD/GeCo를 ROI-1555에 직접 돌린 외부 모델 셀 추가 (Count v1 결과와 병기)
- SAM3 LVIS 48.8 vs 48.5 등 외부 수치 최신판 확인

## 제출 절차 (사용자용 요약)

```text
1. arxiv.org 계정 생성/로그인
2. Submit → category cs.CV (cross-list cs.LG, eess.IV)
3. main.tex + refs.bib + figures/ 업로드 (또는 단일 .zip)
4. 엔도스먼트 면제 여부 확인 → 면제 아니면 엔도서 코드 입력
5. 라이선스 CC BY 4.0 선택 → 메타데이터 입력 → 제출
6. arXiv ID 발급되면: README/index.html/CITATION.cff의 'arXiv pending' → 실제 ID로 교체
   (이 교체는 Claude가 처리 가능)
```
