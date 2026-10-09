# NanoOWL Recycling Detector

Real-time object detection and classification for recycling sorting on NVIDIA Jetson Orin Nano. Uses open-vocabulary vision-language model (OWL-ViT) to classify waste items as recyclable, compostable, or landfill without per-class training.

## Hardware Requirements

- **Jetson Orin Nano** (any variant)
- **JetPack 6.0** (L4T r36.x)
- **USB Webcam** connected to `/dev/video0`
- **4GB+ available memory** (7GB RAM Orin Nano recommended)
- **Docker** with NVIDIA runtime support

## Quick Start

### 1. Verify Hardware

```bash
cat /etc/nv_tegra_release  # Should show: # R36 (release), REVISION: 3.0
docker --version
ls -la /dev/video0         # USB camera should be present
```

### 2. Ensure Docker Group Access

If you get "permission denied" errors:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Build and Setup

```bash
git clone https://github.com/MY_USERNAME/nanoowl-recycle-demo.git
cd nanoowl-recycle-demo
bash setup.sh
```

This will:
- Build the Docker image
- Create persistent engine directory at `~/.nanoowl-demo/engines`
- Pre-build TensorRT engines (5-10 minutes, one-time cost)

### 4. Start the Demo

```bash
bash start.sh
```

Opens dashboard at `http://localhost:7860` automatically. Real-time inference with FPS counter and detection overlays.

### 5. Stop the Demo

```bash
bash stop.sh
```

## Configuration

Edit `config/rules.yaml` to customize:

- **Detection thresholds**: `confidence_threshold`, `nms_threshold`
- **Object categories**: Add/remove items from `categories` (recyclable, compostable, landfill)
- **Alert settings**: Sound, duration, overlay rendering
- **Dashboard FPS**: Stream quality vs. latency trade-off

Changes require container restart: `bash stop.sh && bash start.sh`

### Example: Add Tennis Ball as Recyclable

```yaml
categories:
  recyclable:
    - plastic bottle
    - aluminum can
    - tennis ball  # ← Add this line
```

## Dashboard

The web interface at `http://localhost:7860` shows:

- **Live MJPEG stream** with detection boxes color-coded:
  - 🟢 **Green**: Recyclable (plastics, metals, glass, paper)
  - 🟡 **Yellow**: Compostable (food waste, organics)
  - 🔴 **Red**: Landfill (batteries, styrofoam, vinyl)
- **Real-time FPS counter**
- **Confidence scores** on each detection
- **Category legend**

### Duplicates & Alert Clearing

- **Duplicate box suppression**: Non-maximum suppression (NMS) removes overlapping boxes from same object
- **Alert clearing**: Red alerts automatically clear 3 seconds after object is removed from view

## Offline Operation

Entire model (OWL-ViT weights, TensorRT engines) is baked into the Docker image. Works offline after first start once engines are cached.

### Transferring to Other Jetsons

On the device where image was built:

```bash
# Export image with engines
docker save nanoowl-recycle-demo:latest | gzip > nanoowl-demo.tar.gz

# Transfer to other Jetsons (scp, USB drive, etc)
# Then on target Jetson:
gunzip < nanoowl-demo.tar.gz | docker load
```

## Performance

**Baseline (Orin Nano default clock):**
- Latency: ~200-300ms per frame (inference only)
- FPS: ~3-5 FPS (including rendering)
- GPU Memory: ~3GB (TensorRT optimized)
- CPU: <50% single core

**Optimizations in place:**
- TensorRT FP32 engine (auto-built per device)
- MJPEG quality: 80 (configurable)
- Text embeddings pre-computed
- NMS suppression enabled

## Architecture

```
┌─────────────────────────────────────────────┐
│      FastAPI Dashboard (Port 7860)          │
├─────────────────────────────────────────────┤
│  MJPEG Stream  │  Health Check  │  API      │
├─────────────────────────────────────────────┤
│         NanoOWL (OWL-ViT) Inference         │
│  (Google's open-vocabulary vision model)    │
├─────────────────────────────────────────────┤
│     TensorRT Engine (optimized per device)  │
├─────────────────────────────────────────────┤
│      OpenCV (camera I/O + rendering)        │
├─────────────────────────────────────────────┤
│  USB Webcam (/dev/video0) on Jetson        │
└─────────────────────────────────────────────┘
```

## Files

- `app.py` - FastAPI application with NanoOWL inference loop
- `Dockerfile` - Multi-stage build, OWL-ViT weights baked in
- `config/rules.yaml` - Category definitions and thresholds
- `setup.sh` - Build image + pre-build TensorRT engines
- `start.sh` - Run container with NVIDIA runtime, health check, browser open
- `stop.sh` - Clean shutdown

## Troubleshooting

### Container won't start

```bash
docker logs nanoowl-demo
```

### Camera not detected

```bash
ls -la /dev/video*
```

If missing, plug in USB camera and wait 2-3 seconds.

### Low FPS or high latency

1. Check GPU load: `tegrastats` (should show GPU% increasing during inference)
2. Reduce dashboard FPS in `config/rules.yaml`: `dashboard.fps: 5`
3. Reduce MJPEG quality: Edit `app.py`, find `IMWRITE_JPEG_QUALITY, 80` → lower value

### "permission denied" on docker

```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps  # Should work now
```

## For Workshop Instructors

Each Jetson runs independently. To deploy to 6 devices:

```bash
# On first Jetson (instructor device):
bash setup.sh && bash start.sh

# On other 5 Jetsons:
git clone https://github.com/MY_USERNAME/nanoowl-recycle-demo.git
cd nanoowl-recycle-demo
bash setup.sh  # ~10 min per device in parallel
bash start.sh
```

Students access dashboards via RustDesk at `http://localhost:7860`.

## References

- [NanoOWL (NVIDIA)](https://github.com/NVIDIA-AI-IOT/nanoowl)
- [OWL-ViT (Google)](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
- [Jetson Containers (dusty-nv)](https://github.com/dusty-nv/jetson-containers)

## License

MIT
