FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and training data
COPY app.py .
COPY train.py .
COPY data_ingestion.py .
COPY features.py .
COPY "used_cars (1).csv" .

# Train models during build
RUN mkdir -p models && python train.py

# Expose port (Railway provides $PORT)
EXPOSE 5000

# Run with gunicorn for production
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120