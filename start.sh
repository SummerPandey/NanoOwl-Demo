#!/bin/bash
set -e

echo "================================================"
echo "NanoOWL Recycling Demo - Start"
echo "================================================"

# Check docker access
if ! docker ps &>/dev/null; then
    echo "ERROR: Cannot access Docker. Please run:"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# Check for camera
if [ ! -e /dev/video0 ]; then
    echo "WARNING: /dev/video0 not found. Is the USB camera connected?"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENGINE_DIR="$HOME/.nanoowl-demo/engines"

# Stop any existing container
CONTAINER_NAME="nanoowl-demo"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Ensure engine directory exists
mkdir -p "$ENGINE_DIR"

# Run container
echo "Starting container..."
docker run --detach \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --device /dev/video0 \
    --network host \
    --volume "$ENGINE_DIR:/app/engines" \
    --volume "$SCRIPT_DIR/config:/app/config:ro" \
    --ipc=host \
    nanoowl-recycle-demo:latest

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start container"
    exit 1
fi

echo "✓ Container started: $CONTAINER_NAME"
echo ""
echo "Waiting for health check (this may take 30-40 seconds)..."

# Wait for health check
max_attempts=40
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec "$CONTAINER_NAME" python3 -c "import requests; requests.get('http://localhost:7860/health')" 2>/dev/null; then
        echo "✓ Service is healthy"
        break
    fi
    attempt=$((attempt + 1))
    if [ $((attempt % 10)) -eq 0 ]; then
        echo "  Still waiting... ($attempt/$max_attempts)"
    fi
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    echo "⚠ Health check timeout, but container may still be initializing"
    echo "Check logs with: docker logs $CONTAINER_NAME"
fi

echo ""
echo "================================================"
echo "Dashboard URL: http://localhost:7860"
echo "================================================"
echo ""

# Try to open in browser
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

if command -v xdg-open &>/dev/null; then
    echo "Opening dashboard in browser..."
    xdg-open "http://localhost:7860" 2>/dev/null &
else
    echo "To view dashboard, open browser and navigate to:"
    echo "  http://localhost:7860"
fi

echo ""
echo "Container logs:"
docker logs -f "$CONTAINER_NAME"
