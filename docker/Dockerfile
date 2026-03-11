FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY configs /workspace/configs
COPY docs /workspace/docs

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

CMD ["bash", "-lc", "python scripts/00_env_check.py && sleep infinity"]

