#!/usr/bin/env python3
"""
NanoOWL Recycling Detector - Simple HTTP Server
Real-time object detection with MJPEG streaming dashboard.
"""

import cv2
import numpy as np
import torch
import threading
import time
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
import logging

# Add nanoowl to path
sys.path.insert(0, '/opt/nanoowl')

import yaml
from PIL import Image as PILImage
from nanoowl.owl_predictor import OwlPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, class_name, confidence, bbox, category):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.category = category

COLORS = {
    'recyclable': (0, 255, 0),
    'compostable': (0, 255, 255),
    'landfill': (0, 0, 255),
    'unknown': (128, 128, 128)
}

class RecycleClassifier:
    def __init__(self, config_path='/app/config/rules.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing NanoOWL on {device}...")

        self.predictor = OwlPredictor(
            model_name="google/owlvit-base-patch32",
            device=device
        )

        self.build_embeddings()
        logger.info("RecycleClassifier ready")

    def build_embeddings(self):
        all_classes = []
        self.class_to_category = {}

        for category, classes in self.config['categories'].items():
            for cls in classes:
                all_classes.append(cls)
                self.class_to_category[cls] = category

        self.text_embeddings = self.predictor.encode_text(all_classes)
        self.all_classes = all_classes
        logger.info(f"Encoded {len(all_classes)} classes")

    def classify(self, frame):
        if frame is None or frame.size == 0:
            return []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            pil_image = PILImage.fromarray(frame_rgb)

            predictions = self.predictor.predict(
                image=pil_image,
                text=self.all_classes,
                text_encodings=self.text_embeddings,
                threshold=self.config['detection']['confidence_threshold']
            )

            detections = []

            if predictions is not None and predictions.boxes is not None:
                boxes = predictions.boxes
                scores = predictions.scores
                labels = predictions.labels

                for box, score, label_idx in zip(boxes, scores, labels):
                    score_val = float(score.item() if hasattr(score, 'item') else score)
                    label_idx_val = int(label_idx.item() if hasattr(label_idx, 'item') else label_idx)

                    if score_val >= self.config['detection']['confidence_threshold']:
                        if 0 <= label_idx_val < len(self.all_classes):
                            class_name = self.all_classes[label_idx_val]
                            category = self.class_to_category.get(class_name, 'unknown')

                            x1, y1, x2, y2 = box
                            x1 = float(x1.item() if hasattr(x1, 'item') else x1)
                            y1 = float(y1.item() if hasattr(y1, 'item') else y1)
                            x2 = float(x2.item() if hasattr(x2, 'item') else x2)
                            y2 = float(y2.item() if hasattr(y2, 'item') else y2)

                            if x2 <= 1.0 and y2 <= 1.0:
                                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                            else:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            det = Detection(class_name, score_val, (x1, y1, x2, y2), category)
                            detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return []

    def draw_detections(self, frame, detections):
        output = frame.copy()
        h, w = output.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            color = COLORS.get(det.category, COLORS['unknown'])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} ({det.confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x, text_y = x1, max(20, y1 - 5)

            cv2.rectangle(output, (text_x, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 2), color, -1)
            cv2.putText(output, label, (text_x + 2, text_y - 2), font,
                       font_scale, (255, 255, 255), font_thickness)

        return output

# Global state
classifier = None
camera_frame = None
frame_lock = threading.Lock()
last_detections = []
frame_count = 0
fps_counter = 0

def camera_thread():
    global camera_frame, frame_count

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera: {w}x{h}")

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                camera_frame = frame
                frame_count += 1

        time.sleep(0.03)

def inference_thread():
    global camera_frame, last_detections, classifier

    classifier = RecycleClassifier()

    while True:
        with frame_lock:
            if camera_frame is None:
                time.sleep(0.1)
                continue
            frame = camera_frame.copy()

        try:
            detections = classifier.classify(frame)
            last_detections = detections
        except Exception as e:
            logger.error(f"Inference error: {e}")

        time.sleep(0.05)

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_html().encode())

        elif path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            frame_count = 0
            last_time = time.time()
            fps = 0

            while True:
                with frame_lock:
                    if camera_frame is None:
                        continue
                    frame = camera_frame.copy()

                output = classifier.draw_detections(frame, last_detections)

                frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now

                cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output, f"Detections: {len(last_detections)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')

                time.sleep(0.033)

        elif path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"status": "ok", "classifier_loaded": classifier is not None}
            self.wfile.write(json.dumps(data).encode())

        elif path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {
                "classifier_loaded": classifier is not None,
                "camera_active": camera_frame is not None,
                "detections_count": len(last_detections)
            }
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def get_html():
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

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("NanoOWL Recycling Detector - Starting")
    logger.info("=" * 60)

    # Start threads
    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t1.start()
    t2.start()

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', 7860), RequestHandler)
    logger.info("HTTP Server listening on port 7860")
    server.serve_forever()
