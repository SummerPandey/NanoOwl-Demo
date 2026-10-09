FROM dustynv/nanoowl:r36.3.0

# Base image already has pytorch, opencv, tensorrt, pyyaml, nanoowl.
# app.py only uses the Python standard library on top of those.

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
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start application
CMD ["python3", "-u", "/app/app.py"]
