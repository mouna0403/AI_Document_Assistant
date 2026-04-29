# =========================
# BASE IMAGE
# =========================
FROM python:3.12-slim

# =========================
# SYSTEM DEPENDENCIES
# =========================
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# =========================
# INSTALL UV
# =========================
RUN pip install --no-cache-dir uv

# =========================
# WORKDIR
# =========================
WORKDIR /app

# =========================
# COPY DEPENDENCIES FIRST (cache layer)
# =========================
COPY pyproject.toml uv.lock ./

# install deps via uv
RUN uv sync --frozen

# =========================
# COPY SOURCE CODE
# =========================
COPY src ./src

# =========================
# PYTHON PATH
# =========================
ENV PYTHONPATH=/app/src

# =========================
# EXPOSE FASTAPI PORT
# =========================
EXPOSE 8085

# =========================
# RUN APP (FASTAPI)
# =========================
CMD ["uv", "run", "uvicorn", "project_summarizer.main:app", "--host", "0.0.0.0", "--port", "8085"]
