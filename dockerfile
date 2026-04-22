
# Base image
FROM python:3.12-slim

# Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    build-essential \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies system-wide
RUN uv pip install --system -r pyproject.toml

# Copy source code
COPY src ./src

# Add src to PYTHONPATH so Python can find project_summarizer module
ENV PYTHONPATH=/app/src
# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "src/project_summarizer/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
