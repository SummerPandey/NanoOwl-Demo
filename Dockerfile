FROM dustynv/nanoowl:r36.3.0

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pyyaml==6.0.1 \
    numpy==1.24.3 \
    opencv-python==4.8.0.76 \
    Pillow==10.0.0

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY config/rules.yaml /app/config/

# Create directories for engines and logs
RUN mkdir -p /app/config /app/logs /app/engines

# Expose dashboard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start application
CMD ["python3", "-u", "/app/app.py"]
