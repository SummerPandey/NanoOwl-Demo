#!/bin/bash
set -e

echo "================================================"
echo "NanoOWL Recycling Demo - Setup Script"
echo "================================================"

# Check for docker group access
if ! docker ps &>/dev/null; then
    echo "ERROR: Cannot access Docker. Please run:"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

# Create persistent engine directory
ENGINE_DIR="$HOME/.nanoowl-demo/engines"
mkdir -p "$ENGINE_DIR" "$HOME/.nanoowl-demo/logs"
echo "✓ Created engine directory: $ENGINE_DIR"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "✓ Project directory: $SCRIPT_DIR"

# Check if image already exists
echo ""
IMAGE_TAG="nanoowl-recycle-demo:latest"

if docker image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "✓ Docker image already exists: $IMAGE_TAG"
else
    echo "Building Docker image..."
    docker build \
        --network host \
        --tag "$IMAGE_TAG" \
        --progress=plain \
        "$SCRIPT_DIR"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Image built successfully: $IMAGE_TAG"
    else
        echo "✗ Image build failed"
        exit 1
    fi
fi

# Build the TensorRT image-encoder engine (the big speedup) once per device
echo ""
ENGINE_FILE="owl_image_encoder_patch32.engine"
if [ -f "$ENGINE_DIR/$ENGINE_FILE" ]; then
    echo "✓ TensorRT engine already built: $ENGINE_DIR/$ENGINE_FILE"
else
    echo "Building TensorRT engine (this may take 5-10 minutes)..."
    if docker run --rm \
        --runtime nvidia \
        --network host \
        --volume "$ENGINE_DIR:/engines" \
        --workdir /opt/nanoowl \
        --entrypoint python3 \
        "$IMAGE_TAG" \
        -m nanoowl.build_image_encoder_engine "/engines/$ENGINE_FILE"; then
        echo "✓ TensorRT engine built: $ENGINE_DIR/$ENGINE_FILE"
    else
        echo "✗ TensorRT engine build failed. The demo will still run, but slower (PyTorch)."
    fi
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next, run: bash start.sh"
echo ""
