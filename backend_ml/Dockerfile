# FastAPI Backend Dockerfile
# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code (selective copying to avoid frontend/)
COPY main.py .
COPY models/ ./models/

# Expose port (Render provides PORT env var at runtime)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI server (allow overriding port via PORT env var)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
