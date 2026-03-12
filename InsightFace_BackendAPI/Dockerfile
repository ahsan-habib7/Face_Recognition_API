# ============================================================
# Dockerfile — InsightFace ArcFace + Tesseract OCR API
# ============================================================

FROM python:3.11-slim

# Install system dependencies:
# - tesseract-ocr         : OCR engine
# - tesseract-ocr-ben     : Bengali language pack for NID/Passport text
# - tesseract-ocr-eng     : English language pack (usually default)
# - g++, build-essential  : needed to compile insightface
# - libgl1, libglib2.0-0  : needed by OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download ArcFace model during build (~300MB, cached in volume)
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640,640)); \
print('ArcFace model ready.')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
