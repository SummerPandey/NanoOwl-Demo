#!/bin/bash
set -e

CONTAINER_NAME="nanoowl-demo"

echo "Stopping NanoOWL demo container..."

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    echo "✓ Container stopped and removed"
else
    echo "⚠ Container not running"
fi
