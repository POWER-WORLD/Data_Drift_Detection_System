# ==============================================================================
# Multi-Stage Hardened Dockerfile for Data Drift Detection System
#
# Stage 1: Build & Dependency Compilation
# Stage 2: Minimal, Hardened Non-Root Runtime
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Install temporary build dependencies in builder stage only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to isolated directory prefix
COPY requirements.txt .
RUN pip install --prefix=/install/target --no-warn-script-location -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Hardened Runtime
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Metadata conforming to OCI image specification
LABEL org.opencontainers.image.title="Data Drift Detection System" \
      org.opencontainers.image.description="Enterprise ML Data Drift Detection & Model Monitoring Platform" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="MLOps & Cybersecurity Team" \
      org.opencontainers.image.vendor="DDD System" \
      org.opencontainers.image.licenses="Proprietary" \
      maintainer="ML Ops Team"

# Hardened runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create unprivileged system group and user (non-root UID/GID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -M -d /app appuser

WORKDIR /app

# Copy dependencies from builder stage into system site-packages
COPY --from=builder /install/target /usr/local

# Create operational directories with explicit ownership
RUN mkdir -p /app/data/baseline /app/data/production /app/models /app/reports /app/logs /app/.streamlit && \
    chown -R appuser:appgroup /app

# Copy application assets and code with non-root ownership
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup dashboard/ /app/dashboard/
COPY --chown=appuser:appgroup data/baseline/ /app/data/baseline/
COPY --chown=appuser:appgroup .streamlit/ /app/.streamlit/

# Restrict permissions: read-only code, writable state directories
RUN chmod -R 550 /app/src /app/dashboard /app/.streamlit && \
    chmod -R 770 /app/data/production /app/models /app/reports /app/logs

# Switch to unprivileged non-root user
USER 10001:10001

# Expose internal Streamlit port
EXPOSE 8501

# Self-contained healthcheck using Python standard library (no external binaries needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).getcode() == 200 else 1)"

# Safe startup script: ensure model trained if missing, then launch dashboard
CMD ["sh", "-c", "python -m src.train && streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]
