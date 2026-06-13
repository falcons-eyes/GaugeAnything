# GaugeAnything demo inference server — RTX 5090 (sm_120) / CUDA 13.
# torch cu130 wheel은 CUDA 런타임을 번들하므로 host 드라이버(590)+nvidia runtime만 있으면 된다.
#
# build:  docker build -t gaugeanything:latest .
# run:    docker run -d --name gaugeanything --restart unless-stopped --gpus all -p 8000:8000 \
#           -v $PWD/checkpoints:/app/checkpoints:ro \
#           -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#           gaugeanything:latest
FROM python:3.12-slim

# opencv(libGL/libglib) 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch는 cu130 인덱스로, 나머지는 PyPI로 (레이어 분리 → 캐시 효율)
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r serve/requirements.txt

# 코드 (가중치는 런타임 마운트, 데이터셋/venv는 .dockerignore로 제외)
COPY gaugeanything/ gaugeanything/
COPY experiments/rebar_density_head.py experiments/rebar_density_head.py
COPY serve/ serve/
COPY pyproject.toml ./

ENV PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
