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
| 2026-06-12 | 실행 진위 검증 감사 + E-dyn 설계 | [../VERIFICATION_AUDIT.md](../VERIFICATION_AUDIT.md) · [../DYNAMIC_METROLOGY_DESIGN.md](../DYNAMIC_METROLOGY_DESIGN.md) | 결정적 재현 3건 전부 일치 · ad-hoc 1건 정식화 · 동적 환경 5개 데이터 실증 검증 |
| 2026-06-12 | ADT ATEK access/depth upper-bound — E-dyn-3a/b/c | [2026-06-12_adt-atek-access-probe.md](2026-06-12_adt-atek-access-probe.md) | ADT signed URL 확보 · 3D→2D IoU 0.675 · box-only 25.4%(한계) · oracle depth fusion 8.7%(2 seq/480f/229 obj), 0.5m/s+도 9.1% · ROI-only 316% 붕괴 |
| 2026-06-12 | Physical AI coverage P0-P2 | [2026-06-12_physical-coverage-p0-p2.md](2026-06-12_physical-coverage-p0-p2.md) · [../PHYSICAL_COVERAGE_MATRIX.md](../PHYSICAL_COVERAGE_MATRIX.md) | 15 coverage atoms · project page 12-case gallery · adapter sprint queue(SmartDoc/MIDV→TimberSeg→DeepFish→BOP→KITTI) |
| 2026-06-12 | Owned model track — GaugeHead-Tiny | [2026-06-12_owned-model-gaugehead-tiny.md](2026-06-12_owned-model-gaugehead-tiny.md) · [../MODEL_RESEARCH_ROADMAP.md](../MODEL_RESEARCH_ROADMAP.md) | 자체 measurement head 첫 합격: rel.err 0.4724 vs quantile 0.4804 vs neural M2 0.564 · worst-source 0.7201 한계 명시 |
| 2026-06-13 | Codex → Claude 최신 인계 | [2026-06-13_codex-to-claude-handoff.md](2026-06-13_codex-to-claude-handoff.md) | P0-P2 coverage · SmartDoc 5k scale smoke result · GaugeHead-Tiny 0.4724 · Spark artifact 경로와 다음 작업 |
| 2026-06-13 | M2 v2-b conformal interval | [2026-06-13_m2-v2b-uncertainty-conformal.md](2026-06-13_m2-v2b-uncertainty-conformal.md) | rel err 0.4724 유지 + 전 source coverage ≥0.90(log cross-conformal) · adaptive 방법 CrackTree200 붕괴(0.21/0.11) · concept shift라 OOD 신호 3종 플래깅 실패(정직한 음성) |
| 2026-06-11 | E-mm-3 krkCMd profile-level μm 검증 | [2026-06-11_krkcmd-profile-emm3.md](2026-06-11_krkcmd-profile-emm3.md) | 물리 폭 GT 19,098 profiles · DLM 11.1μm · GaugeProfile+cal 25.9μm ≈ AED 26.5μm |
| 2026-06-11 | Codex handoff 기록 | [2026-06-11_codex-handoff.md](2026-06-11_codex-handoff.md) | Claude 재개용 상세 인계 · 로컬/Spark 경로 · 새 산출물/검증/다음 작업 |
| 2026-06-11 | 객관적 연구 감사 + 다음 실험 계획 | [2026-06-11_objective-audit-next-experiments.md](2026-06-11_objective-audit-next-experiments.md) | krkCMd 5-fold 27.8±2.5μm · series holdout worst 46.7μm · 모델/데이터/학습 로드맵 |
| 2026-06-11 | E-cnt-2 rebar SAHI-style tiled SAM3 | [2026-06-11_rebar-sahi-ecnt2.md](2026-06-11_rebar-sahi-ecnt2.md) | global MAE 13.2→SAHI 7.35 · acc@10% 0→20% · dense undercount remains |

이전 단계들의 결과 요약은 [../../experiments/RESULTS.md](../../experiments/RESULTS.md),
감사 자체는 [../RIGOR_AUDIT.md](../RIGOR_AUDIT.md) 참조.
