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
mkdir -p "$ENGINE_DIR"
echo "✓ Created engine directory: $ENGINE_DIR"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "✓ Project directory: $SCRIPT_DIR"

# Build Docker image
echo ""
echo "Building Docker image..."
IMAGE_TAG="nanoowl-recycle-demo:latest"

docker build \
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

# Create a temporary container to build TensorRT engines
echo ""
echo "Building TensorRT engines (this may take 5-10 minutes)..."

TEMP_CONTAINER="nanoowl-engine-builder-$$"

docker run --rm \
    --name "$TEMP_CONTAINER" \
    --runtime nvidia \
    --volume "$ENGINE_DIR:/engines" \
    --entrypoint python3 \
    "$IMAGE_TAG" \
    -c "
from nanoowl.predictor import OwlPredictor
import os
os.chdir('/engines')
print('Initializing OWL predictor with TensorRT engine building...')
predictor = OwlPredictor(model_name='google/owlvit-base-patch32', device='cuda')
print('✓ TensorRT engines built and cached in /engines')
"

if [ $? -eq 0 ]; then
    echo "✓ TensorRT engines built successfully"
    echo ""
    echo "Engine files in $ENGINE_DIR:"
    ls -lah "$ENGINE_DIR" 2>/dev/null || echo "  (engines directory)"
else
    echo "⚠ Warning: Engine building had issues, but may still work"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next, run: bash start.sh"
echo ""
