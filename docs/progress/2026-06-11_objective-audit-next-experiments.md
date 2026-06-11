# 2026-06-11 — 객관적 연구 감사 + 다음 실험 계획

상세 문서: [../RESEARCH_AUDIT_AND_NEXT_EXPERIMENTS.md](../RESEARCH_AUDIT_AND_NEXT_EXPERIMENTS.md)

## 왜 했나
홍보 페이지 정리 후, 다시 연구 본질로 돌아와 현재 결과가 어디까지 방어 가능하고 어디부터는
과장/체리피킹 위험이 있는지 점검했다. 특히 krkCMd `25.9μm`가 단일 split 우연인지 확인하기 위해
추가 robustness audit를 수행했다.

## 새 실험: krkCMd split robustness audit

스크립트:

```bash
.venv/bin/python experiments/krkcmd_split_audit.py
```

결과:

| Split audit | DLM author | AED author | GaugeProfile uncal | GaugeProfile+cal |
|---|---:|---:|---:|---:|
| group 5-fold MAE mean±std | 14.0±2.4μm | 34.4±5.4μm | 35.9±3.1μm | **27.8±2.5μm** |
| leave-one-stage mean±std | 13.9±1.9μm | 34.1±2.3μm | 35.6±1.4μm | **27.7±2.1μm** |
| leave-one-series mean±std | 16.7±9.2μm | 38.6±13.1μm | 39.0±10.0μm | **30.7±9.9μm** |
| leave-one-series worst | 32.2μm | 56.8μm | 54.2μm | **46.7μm** |

## 판정
- 기존 E-mm-3 단일 split `25.9μm`는 cherry-picked miracle은 아니다. group 5-fold 평균이
  `27.8±2.5μm`로 안정적이다.
- 그러나 leave-one-series worst `46.7μm`로 domain/series shift가 있다.
- camera-ready 표현은 `25.9μm` 단독보다 `group 5-fold 27.8±2.5μm; series holdout worst 46.7μm`
  같이 쓰는 것이 더 정직하다.

## 다음 우선순위
1. SAHI rebar counting: E-cnt-1 실패가 global scale/crowding 때문인지 확인.
2. T-LESS GT-mask measurement upper bound: segmentation이 완벽할 때 CAD+pose mm 유도 검증.
3. krkCMd image subset extraction: profile-level에서 image-level promptable measurement로 확장.
4. M2 v2: domain/scale-conditioned width calibration.
