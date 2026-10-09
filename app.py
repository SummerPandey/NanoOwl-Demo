#!/usr/bin/env python3
"""
NanoOWL Recycling Detector - FastAPI Dashboard
Real-time object detection with classification overlay on MJPEG stream.
"""

import cv2
import numpy as np
import torch
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

# Add nanoowl to path
sys.path.insert(0, '/opt/nanoowl')

from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import yaml
from PIL import Image as PILImage

from nanoowl.owl_predictor import OwlPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized or pixel)
    category: str  # 'recyclable', 'compostable', 'landfill'

# Color mapping for categories (BGR format for OpenCV)
COLORS = {
    'recyclable': (0, 255, 0),    # Green
    'compostable': (0, 255, 255), # Yellow
    'landfill': (0, 0, 255),      # Red
    'unknown': (128, 128, 128)    # Gray
}

class RecycleClassifier:
    def __init__(self, config_path: str = '/app/config/rules.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loading config from {config_path}")

        # Initialize OWL predictor with CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing NanoOWL predictor on {device}...")

        self.predictor = OwlPredictor(
            model_name="google/owlvit-base-patch32",
            device=device
        )

        # Build category text embeddings
        self.build_embeddings()

        # Tracking for detections
        self.current_detections = {}
        self.frame_count = 0

        logger.info("RecycleClassifier initialized")

    def build_embeddings(self):
        """Pre-compute text embeddings for all categories."""
        logger.info("Building text embeddings for categories...")

        all_classes = []
        self.class_to_category = {}

        for category, classes in self.config['categories'].items():
            for cls in classes:
                all_classes.append(cls)
                self.class_to_category[cls] = category

        # Encode all class names to embeddings
        self.text_embeddings = self.predictor.encode_text(all_classes)
        self.all_classes = all_classes
        logger.info(f"Built embeddings for {len(all_classes)} classes")

    def classify(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on frame and return detections."""
        if frame is None or frame.size == 0:
            return []

        try:
            # Convert BGR to RGB for PIL/transformers
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Convert to PIL Image for OWL predictor
            pil_image = PILImage.fromarray(frame_rgb)

            # Run inference with pre-computed embeddings
            predictions = self.predictor.predict(
                image=pil_image,
                text=self.all_classes,
                text_encodings=self.text_embeddings,
                threshold=self.config['detection']['confidence_threshold']
            )

            detections = []

            if predictions is not None:
                # predictions is OwlDecodeOutput with:
                # - boxes: tensor of shape (N, 4) with normalized coords [x1, y1, x2, y2]
                # - scores: tensor of shape (N,)
                # - labels: tensor of shape (N,) with class indices
                # - input_indices: tensor of shape (N,)

                boxes = predictions.boxes
                scores = predictions.scores
                labels = predictions.labels

                if boxes is not None and len(boxes) > 0:
                    for box, score, label_idx in zip(boxes, scores, labels):
                        score_val = float(score.item()) if hasattr(score, 'item') else float(score)
                        label_idx_val = int(label_idx.item()) if hasattr(label_idx, 'item') else int(label_idx)

                        if score_val >= self.config['detection']['confidence_threshold']:
                            if 0 <= label_idx_val < len(self.all_classes):
                                class_name = self.all_classes[label_idx_val]
                                category = self.class_to_category.get(class_name, 'unknown')

                                # Convert normalized coords to pixel coords
                                x1, y1, x2, y2 = box
                                x1 = float(x1.item() if hasattr(x1, 'item') else x1)
                                y1 = float(y1.item() if hasattr(y1, 'item') else y1)
                                x2 = float(x2.item() if hasattr(x2, 'item') else x2)
                                y2 = float(y2.item() if hasattr(y2, 'item') else y2)

                                # Normalize if needed (OWL outputs normalized coords)
                                if x2 <= 1.0 and y2 <= 1.0:
                                    x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                                else:
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                                det = Detection(
                                    class_name=class_name,
                                    confidence=score_val,
                                    bbox=(x1, y1, x2, y2),
                                    category=category
                                )
                                detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.category, COLORS['unknown'])

            # Clamp coordinates
            h, w = output.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label with background
            label = f"{det.class_name} ({det.confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x, text_y = x1, max(20, y1 - 5)

            # Background rectangle
            cv2.rectangle(
                output,
                (text_x, text_y - text_size[1] - 4),
                (text_x + text_size[0] + 4, text_y + 2),
                color,
                -1
            )

            # Text
            cv2.putText(
                output,
                label,
                (text_x + 2, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )

        return output

# Global state
app = FastAPI(title="NanoOWL Recycling Detector")
classifier = None
camera_frame = None
camera_lock = asyncio.Lock()
frame_time = time.time()
last_detections: List[Detection] = []

def init_classifier():
    global classifier
    if classifier is None:
        classifier = RecycleClassifier()

async def read_frames():
    """Continuously read frames from camera."""
    global camera_frame, frame_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera at /dev/video0")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera opened: {w}x{h} @ {fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            await asyncio.sleep(0.1)
            continue

        async with camera_lock:
            camera_frame = frame.copy()
            frame_time = time.time()

        await asyncio.sleep(0.033)  # ~30 FPS read rate

async def inference_loop():
    """Run continuous inference on camera frames."""
    global camera_frame, last_detections

    init_classifier()
    logger.info("Inference loop started")

    frame_count = 0
    last_log_time = time.time()

    while True:
        async with camera_lock:
            if camera_frame is None:
                await asyncio.sleep(0.1)
                continue

            frame = camera_frame.copy()

        try:
            start = time.time()
            detections = classifier.classify(frame)
            latency = (time.time() - start) * 1000

            last_detections = detections
            frame_count += 1

            # Log periodically
            now = time.time()
            if now - last_log_time >= 5.0:
                if detections:
                    classes = [d.class_name for d in detections]
                    logger.info(f"Detected ({frame_count}): {classes} (latency: {latency:.1f}ms)")
                frame_count = 0
                last_log_time = now

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)

        await asyncio.sleep(0.05)  # ~20 FPS inference

async def mjpeg_generator():
    """Generate MJPEG stream with annotations."""
    global camera_frame, last_detections

    init_classifier()

    frame_count = 0
    last_time = time.time()
    fps_display = 0

    while True:
        async with camera_lock:
            if camera_frame is None:
                await asyncio.sleep(0.033)
                continue

            frame = camera_frame.copy()

        try:
            # Draw last detections on frame
            output = classifier.draw_detections(frame, last_detections)

            # Add FPS counter
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps_display = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            cv2.putText(
                output,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Detection count
            cv2.putText(
                output,
                f"Detections: {len(last_detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # MJPEG encoding
            ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                await asyncio.sleep(0.033)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')

            await asyncio.sleep(0.033)  # ~30 FPS MJPEG

        except Exception as e:
            logger.error(f"MJPEG generation error: {e}", exc_info=True)
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("Starting NanoOWL Recycling Detector")
    logger.info("=" * 60)

    # Start camera reader task
    asyncio.create_task(read_frames())

    # Start inference task
    asyncio.create_task(inference_loop())

@app.get("/")
async def root():
    """Serve HTML dashboard."""
    return HTMLResponse(html_dashboard())

@app.get("/video_feed")
async def video_feed():
    """Stream MJPEG video."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/stats")
async def stats():
    """Return detection statistics."""
    return {
        "classifier_loaded": classifier is not None,
        "camera_active": camera_frame is not None,
        "detections_count": len(last_detections),
        "timestamp": datetime.now().isoformat()
    }

def html_dashboard():
    """Generate HTML dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NanoOWL Recycling Detector</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                width: 100%;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 28px;
            }
            .video-container {
                width: 100%;
                background: #000;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 30px;
                aspect-ratio: 4/3;
            }
            #video {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-label {
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .stat-value {
                font-size: 20px;
                font-weight: bold;
                color: #333;
            }
            .legend {
                display: flex;
                justify-content: center;
                gap: 30px;
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .legend-box {
                width: 20px;
                height: 20px;
                border-radius: 4px;
            }
            .legend-label {
                font-size: 14px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>♻️ NanoOWL Recycling Detector</h1>
            <div class="video-container">
                <img id="video" src="/video_feed" alt="Live video stream">
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Status</div>
                    <div class="stat-value" id="status">Ready</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Detections</div>
                    <div class="stat-value" id="detections">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Camera</div>
                    <div class="stat-value" id="camera">Waiting</div>
                </div>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background: #00FF00;"></div>
                    <div class="legend-label">Recyclable</div>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #FFFF00;"></div>
                    <div class="legend-label">Compostable</div>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #FF0000;"></div>
                    <div class="legend-label">Landfill</div>
                </div>
            </div>
        </div>
        <script>
            async function updateStats() {
                try {
                    const res = await fetch('/api/stats');
                    const data = await res.json();
                    document.getElementById('status').textContent =
                        data.classifier_loaded ? 'Processing' : 'Initializing';
                    document.getElementById('camera').textContent =
                        data.camera_active ? 'Active' : 'Waiting';
                    document.getElementById('detections').textContent =
                        data.detections_count || '0';
                } catch (e) {
                    console.error('Stats error:', e);
                }
            }
            setInterval(updateStats, 1000);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
