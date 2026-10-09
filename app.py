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
from typing import Dict, List, Tuple
from dataclasses import dataclass

from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import yaml

from nanoowl.predictor import OwlPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    category: str  # 'recyclable', 'compostable', 'landfill'

# Color mapping for categories
COLORS = {
    'recyclable': (0, 255, 0),    # Green (BGR)
    'compostable': (0, 255, 255), # Yellow (BGR)
    'landfill': (0, 0, 255),      # Red (BGR)
    'unknown': (128, 128, 128)    # Gray (BGR)
}

class RecycleClassifier:
    def __init__(self, config_path: str = '/app/config/rules.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loading config from {config_path}")

        # Initialize OWL predictor
        logger.info("Initializing NanoOWL predictor...")
        self.predictor = OwlPredictor(
            model_name="google/owlvit-base-patch32",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build category text embeddings
        self.build_embeddings()

        # Tracking for alerts
        self.last_alert_time = defaultdict(float)
        self.current_detections = {}
        self.detection_history = []

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

        # Use OWL's text encoder
        self.text_embeddings = self.predictor.encode_text(all_classes)
        self.all_classes = all_classes
        logger.info(f"Built embeddings for {len(all_classes)} classes")

    def classify(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on frame and return detections."""
        if frame is None or frame.size == 0:
            return []

        # OWL predictor expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict (returns list of predictions with scores)
        predictions = self.predictor.predict(
            image=frame_rgb,
            text=self.all_classes,
            threshold=self.config['detection']['confidence_threshold']
        )

        detections = []

        if predictions is not None and len(predictions) > 0:
            # predictions is a dict with 'boxes', 'scores', 'class_ids'
            if isinstance(predictions, dict):
                boxes = predictions.get('boxes', [])
                scores = predictions.get('scores', [])
                class_ids = predictions.get('class_ids', [])
            else:
                # Fallback if predictions is a different format
                return []

            # Filter by confidence and convert to Detection objects
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= self.config['detection']['confidence_threshold']:
                    if 0 <= class_id < len(self.all_classes):
                        class_name = self.all_classes[int(class_id)]
                        category = self.class_to_category.get(class_name, 'unknown')

                        # Normalize box to frame coordinates if needed
                        x1, y1, x2, y2 = box
                        if x2 <= 1.0:  # Normalized coords
                            h, w = frame.shape[:2]
                            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                        else:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        det = Detection(
                            class_name=class_name,
                            confidence=float(score),
                            bbox=(x1, y1, x2, y2),
                            category=category
                        )
                        detections.append(det)

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.category, COLORS['unknown'])

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label with background
            label = f"{det.class_name} ({det.confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x, text_y = x1, y1 - 5

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

def init_classifier():
    global classifier
    if classifier is None:
        classifier = RecycleClassifier()

async def read_frames():
    """Continuously read frames from camera."""
    global camera_frame, frame_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            await asyncio.sleep(0.1)
            continue

        async with camera_lock:
            camera_frame = frame.copy()
            frame_time = time.time()

        await asyncio.sleep(0.033)  # ~30 FPS

async def inference_loop():
    """Run continuous inference on camera frames."""
    global camera_frame

    init_classifier()

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

            # Log detections
            if detections:
                classes = [d.class_name for d in detections]
                logger.info(f"Detected: {classes} (latency: {latency:.1f}ms)")

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)

        await asyncio.sleep(0.1)

async def mjpeg_generator():
    """Generate MJPEG stream with annotations."""
    global camera_frame

    init_classifier()

    frame_count = 0
    last_time = time.time()
    fps_display = 0

    while True:
        async with camera_lock:
            if camera_frame is None:
                await asyncio.sleep(0.1)
                continue

            frame = camera_frame.copy()

        try:
            # Run inference
            detections = classifier.classify(frame)

            # Draw detections
            output = classifier.draw_detections(frame, detections)

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

            # MJPEG encoding
            ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')

            await asyncio.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logger.error(f"MJPEG generation error: {e}", exc_info=True)
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("Starting NanoOWL Recycling Detector")

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
                    <div class="stat-value" id="camera">Active</div>
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
