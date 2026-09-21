FROM python:3.11-slim

# System dependencies: tesseract for OCR, cmake+build tools for dlib
# (face_recognition's dependency) to compile during pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
