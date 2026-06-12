# Progress Log — 단계별 진행 기록

> 각 진행 단계를 검토 가능한 MD로 남긴다. 파일명: `YYYY-MM-DD_<step>.md`
> 규약: 단계마다 (1) 무엇을 왜 했나 (2) 공식 수치(프로토콜 명시) (3) 발견 (4) 다음 단계.

| 날짜 | 단계 | 파일 | 핵심 결과 |
|---|---|---|---|
| 2026-06-11 | 엄밀성 감사 + 수정 배치 + 후속 2종 | [2026-06-11_audit-fixes.md](2026-06-11_audit-fixes.md) | 2.44× 공식화 · matting 실전이 실패(정직) · 앙상블 구조 0→0.37 · 틸트 19%→0.7% |
| 2026-06-11 | OSS 분리 + matting v2 + 라우터 통합 | [2026-06-11_oss-migration-matting-v2.md](2026-06-11_oss-migration-matting-v2.md) | 공개 repo 전환 · v2 실 fray 0.949(>고전 0.860) · PlaneScale/앙상블 라우터 |
| 2026-06-11 | 사업자료 패키지 + 실측 mm GT 체계 | [2026-06-11_biz-package-capture-protocol.md](2026-06-11_biz-package-capture-protocol.md) | 캡처 프로토콜+인쇄보드(검출 0.38%)+캘리퍼 평가기 · 카운팅은 kaggle 토큰 블로커 |
| 2026-06-11 | mm GT 대체 확보 + E-mm-1/E-cnt-1 | [2026-06-11_metric-substitutes-emm1-ecnt1.md](2026-06-11_metric-substitutes-emm1-ecnt1.md) | 동전 LOO 1.74%(±5% 100%) · rebar zero-shot 능력갭(6프롬프트 실패) · 논문 자료 2종 |
| 2026-06-12 | 감사 후속 D·N1b·N2·N3 | [2026-06-12_audit-followups-n2-n3.md](2026-06-12_audit-followups-n2-n3.md) | SAM3 로짓 노출 확정 · 분위보정 0.480>신경 0.564 격하 · plane-scale 상한 2.83% · SAHI 8.9 |
| 2026-06-12 | promptable mm 체인 3대 진전 | [2026-06-12_promptable-mm-chain.md](2026-06-12_promptable-mm-chain.md) | E-mm-2b 부품 2.5%≈상한 · M2v2-a 0.437 합격선 돌파 · E-mm-3b 체인 개통+θ* 전이 −30% |
| 2026-06-12 | 폭 병목 해체 — mask=WHERE signal=WIDTH | [2026-06-12_width-bottleneck-resolved.md](2026-06-12_width-bottleneck-resolved.md) | 물리적 불가 아님 확정 · 1D CNN 신호폭 23~40μm(조건부, mask 대비 4~6×) · 잔여=위치 커버리지 |
| 2026-06-12 | SpatialClaw 검토 → E-loop 사다리 → 멀티-인스턴스 | [2026-06-12_agentic-loop-ladder.md](2026-06-12_agentic-loop-ladder.md) | recall 93%/30.5μm — 커버리지 천장은 문제정의 오류 · VLM 에이전트 현 단계 불필요 판정 |
| 2026-06-11 | E-mm-3 krkCMd profile-level μm 검증 | [2026-06-11_krkcmd-profile-emm3.md](2026-06-11_krkcmd-profile-emm3.md) | 물리 폭 GT 19,098 profiles · DLM 11.1μm · GaugeProfile+cal 25.9μm ≈ AED 26.5μm |
| 2026-06-11 | Codex handoff 기록 | [2026-06-11_codex-handoff.md](2026-06-11_codex-handoff.md) | Claude 재개용 상세 인계 · 로컬/Spark 경로 · 새 산출물/검증/다음 작업 |
| 2026-06-11 | 객관적 연구 감사 + 다음 실험 계획 | [2026-06-11_objective-audit-next-experiments.md](2026-06-11_objective-audit-next-experiments.md) | krkCMd 5-fold 27.8±2.5μm · series holdout worst 46.7μm · 모델/데이터/학습 로드맵 |
| 2026-06-11 | E-cnt-2 rebar SAHI-style tiled SAM3 | [2026-06-11_rebar-sahi-ecnt2.md](2026-06-11_rebar-sahi-ecnt2.md) | global MAE 13.2→SAHI 7.35 · acc@10% 0→20% · dense undercount remains |

이전 단계들의 결과 요약은 [../../experiments/RESULTS.md](../../experiments/RESULTS.md),
감사 자체는 [../RIGOR_AUDIT.md](../RIGOR_AUDIT.md) 참조.
