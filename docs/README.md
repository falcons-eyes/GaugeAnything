# GaugeAnything — 프로젝트 페이지

Academic Project Page Template(Bulma) 기반. 실측 결과·정성 갤러리·draw.io 아키텍처 도식 포함.

## 로컬 실행

```bash
cd GaugeAnything/docs
python3 -m http.server 8848
# 브라우저에서 http://localhost:8848 열기
```

## 구성
- `index.html` — 단일 페이지 (CDN: Bulma + bulma-carousel + FontAwesome + Academicons)
- `assets/architecture.svg` — 아키텍처 도식 (페이지 임베드용)
- `assets/architecture.drawio` — 편집 가능한 원본 (draw.io / diagrams.net에서 열기)
- `assets/gallery_*.png` — SAM3 정성 결과 (CrackSeg9k 소스별, `experiments/gauge_gallery.py`로 재생성)
- `MODEL_RESEARCH_ROADMAP.md` — 자체 GaugeHead/GaugeSpecialist 모델 연구 로드맵과 첫 tiny specialist 결과
- `PHYSICAL_COVERAGE_MATRIX.md` — physical quantity coverage matrix

## 갤러리 재생성 (Spark)
```bash
python experiments/gauge_gallery.py --out docs/assets
```
