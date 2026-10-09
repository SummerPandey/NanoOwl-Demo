FROM dustynv/nanoowl:r36.3.0

# Base image already has pytorch, torchvision, opencv, tensorrt, pyyaml
# Install additional web framework dependencies
RUN pip install fastapi uvicorn[standard] python-multipart --no-cache-dir || \
    echo "Note: pip packages may already be available in base image"

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
