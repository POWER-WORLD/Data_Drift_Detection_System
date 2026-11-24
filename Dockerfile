FROM python:3.10-slim

LABEL maintainer="ML Ops Team"
LABEL description="Data Drift Detection System for ML Models"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY dashboard/ dashboard/
COPY data/baseline/ data/baseline/
COPY .streamlit/ .streamlit/

# Create necessary directories
RUN mkdir -p data/production models reports logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Train model and start dashboard
CMD ["sh", "-c", "python -m src.train && streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]
