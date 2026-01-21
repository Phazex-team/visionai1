#!/usr/bin/env python3
"""
Flask Preview Server for Retail Fraud Detection - NEW UNIFIED FRAMEWORK
Uses DetectionPipeline, ModelFactory, OptimizationManager, POSProcessor, EvidenceRecorder
Streams video with detection overlays in headless mode.
Access via browser at http://localhost:5000
"""
import sys
import os
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify

sys.path.insert(0, os.path.dirname(__file__))

# New unified framework imports
from config_models import ApplicationConfig, DetectionConfig, OptimizationConfig, POSConfig, EvidenceConfig, ZoneConfig, FaceMaskingConfig
from detection_pipeline import DetectionPipeline, load_config_from_file
from model_factory import ModelFactory
from face_masking import get_face_masker, FaceMaskingConfig as FMConfig, reset_face_masker

# Global state
app = Flask(__name__)
face_masker = None
current_frame = None
frame_lock = threading.Lock()
detection_pipeline = None
detection_thread = None

stats = {
    'model': 'None',
    'fps': 0.0,
    'detections': 0,
    'frame_num': 0,
    'total_frames': 0,
    'fraud_events': 0,
    'inference_time_ms': 0.0,
    'pos_matches': 0,
    'pos_mismatches': 0
}

# HTML template (simplified for unified framework)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Retail Fraud Detection - Unified Framework</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #1a1a2e; 
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 { 
            font-size: 1.5em; 
            color: #e94560;
        }
        .main-container {
            display: flex;
            padding: 20px;
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .video-section { flex: 2; }
        .video-container {
            background: #0f0f23;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        .video-container img { width: 100%; display: block; }
        .stats-bar {
            background: #16213e;
            padding: 15px 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 10px;
        }
        .stat-item { text-align: center; }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #e94560;
        }
        .stat-label {
            font-size: 0.85em;
            color: #888;
            text-transform: uppercase;
        }
        .sidebar { flex: 1; max-width: 400px; }
        .panel {
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .panel-header {
            background: #0f3460;
            padding: 12px 15px;
            font-weight: bold;
        }
        .panel-body {
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }
        .info-row:last-child { border-bottom: none; }
        .info-label { color: #888; }
        .info-value { color: #2196F3; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛒 Retail Fraud Detection - Unified Framework</h1>
        <div style="color: #888;">Powered by DetectionPipeline</div>
    </div>
    
    <div class="main-container">
        <div class="video-section">
            <div class="video-container">
                <img src="/video_feed" alt="Video Stream">
                <div class="stats-bar">
                    <div class="stat-item">
                        <div class="stat-value" id="fps">0.0</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="detections">0</div>
                        <div class="stat-label">Detections</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="frame">0/0</div>
                        <div class="stat-label">Frame</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="frauds">0</div>
                        <div class="stat-label">Fraud Events</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <div class="panel-header">� Select Files</div>
                <div class="panel-body">
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em;">🎬 Video File:</label>
                        <select id="videoSelect" style="width: 100%; padding: 8px; border-radius: 5px; background: #0f3460; color: #eee; border: 1px solid #333; cursor: pointer;">
                            <option value="">-- Select a video --</option>
                        </select>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em;">📋 POS File (XML):</label>
                        <select id="posSelect" style="width: 100%; padding: 8px; border-radius: 5px; background: #0f3460; color: #eee; border: 1px solid #333; cursor: pointer;">
                            <option value="">-- Optional --</option>
                        </select>
                    </div>
                    <button onclick="startProcessing()" style="width: 100%; padding: 10px; background: #e94560; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; margin-top: 10px;">▶️ Start Processing</button>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">�📊 Pipeline Status</div>
                <div class="panel-body">
                    <div class="info-row">
                        <span class="info-label">Model:</span>
                        <span class="info-value" id="modelName">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Inference Time:</span>
                        <span class="info-value" id="inferenceTime">0ms</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Processed Frames:</span>
                        <span class="info-value" id="processedFrames">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Total Detections:</span>
                        <span class="info-value" id="totalDetections">0</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">📋 POS Integration</div>
                <div class="panel-body">
                    <div class="info-row">
                        <span class="info-label">POS Matches:</span>
                        <span class="info-value" id="posMatches">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">POS Mismatches:</span>
                        <span class="info-value" id="posMismatches">0</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">📁 Evidence Recording</div>
                <div class="panel-body">
                    <div class="info-row">
                        <span class="info-label">Status:</span>
                        <span class="info-value" id="evidenceStatus">Active</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Output Dir:</span>
                        <span class="info-value" style="font-size: 0.8em;" id="evidenceDir">-</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">ℹ️ Framework Info</div>
                <div class="panel-body">
                    <p style="font-size: 0.9em; color: #888;">
                        ✅ Unified configuration<br>
                        ✅ Centralized optimization<br>
                        ✅ POS data integration<br>
                        ✅ Evidence recording<br>
                        ✅ GPU memory management
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Load available files on page load
        function loadFiles() {
            fetch('/get_files')
                .then(r => r.json())
                .then(data => {
                    // Populate video select
                    const videoSelect = document.getElementById('videoSelect');
                    data.videos.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v;
                        opt.textContent = v;
                        videoSelect.appendChild(opt);
                    });
                    
                    // Populate POS select
                    const posSelect = document.getElementById('posSelect');
                    data.pos_files.forEach(p => {
                        const opt = document.createElement('option');
                        opt.value = p;
                        opt.textContent = p;
                        posSelect.appendChild(opt);
                    });
                })
                .catch(e => console.error('Error loading files:', e));
        }
        
        // Start processing with selected files
        function startProcessing() {
            const video = document.getElementById('videoSelect').value;
            const pos = document.getElementById('posSelect').value;
            
            if (!video) {
                alert('Please select a video file');
                return;
            }
            
            fetch('/start_process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video: video, pos: pos})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    alert('Processing started: ' + video);
                }
            })
            .catch(e => alert('Error: ' + e));
        }
        
        function updateStats() {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('frame').textContent = data.frame_num + '/' + data.total_frames;
                    document.getElementById('frauds').textContent = data.fraud_events;
                    document.getElementById('modelName').textContent = data.model;
                    document.getElementById('inferenceTime').textContent = data.inference_time_ms.toFixed(1) + 'ms';
                    document.getElementById('posMatches').textContent = data.pos_matches;
                    document.getElementById('posMismatches').textContent = data.pos_mismatches;
                });
        }
        
        // Load files and start stats updates on page load
        loadFiles();
        setInterval(updateStats, 500);
    </script>
</body>
</html>
'''

@app.route('/get_files')
def get_files():
    """Get list of video and POS files from videos folder"""
    videos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos')
    
    video_files = []
    pos_files = []
    
    try:
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(f)
                elif f.endswith(('.xml', '.csv')):
                    pos_files.append(f)
    except Exception as e:
        print(f"Error listing files: {e}")
    
    return jsonify({
        'videos': sorted(video_files),
        'pos_files': sorted(pos_files)
    })

@app.route('/start_process', methods=['POST'])
def start_process():
    """Start processing with selected video and POS file"""
    from flask import request
    data = request.json
    video_file = data.get('video', '')
    pos_file = data.get('pos', '')
    
    if not video_file:
        return jsonify({'error': 'No video selected'}), 400
    
    # Update config
    videos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos')
    video_path = os.path.join(videos_dir, video_file)
    
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found: {video_path}'}), 400
    
    # Update global pipeline with new video
    global detection_pipeline, detection_thread
    
    try:
        # Load or create config
        from config_models import DataFormat
        config = ApplicationConfig(
            video_path=video_path,
            detection=DetectionConfig(model_name='yoloworld'),
            optimization=OptimizationConfig(model_name='yoloworld'),
            pos=POSConfig(
                enabled=True if pos_file else False,
                xml_path=os.path.join(videos_dir, pos_file) if pos_file else '',
                data_format=DataFormat.XML
            ),
            evidence=EvidenceConfig(enabled=True)
        )
        
        # Start detection in background thread (pass config, not pipeline)
        if detection_thread and detection_thread.is_alive():
            print("Stopping previous detection thread...")
        
        detection_thread = threading.Thread(target=run_detection_loop, args=(config,), daemon=True)
        detection_thread.start()
        
        return jsonify({'status': 'started', 'video': video_file, 'pos': pos_file})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Stream MJPEG video"""
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    _, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                else:
                    black = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black, "Waiting for video...", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    _, buffer = cv2.imencode('.jpg', black)
                    frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

def update_frame(frame, frame_num=None):
    """Update displayed frame with face masking"""
    global current_frame, face_masker
    
    if face_masker:
        display_frame = face_masker.mask_faces(frame, frame_num=frame_num)
    else:
        display_frame = frame.copy()
    
    with frame_lock:
        current_frame = display_frame

def run_detection_loop(config: ApplicationConfig):
    """Detection pipeline loop"""
    global detection_pipeline, stats
    
    print(f"\n{'='*70}")
    print(f"STARTING DETECTION PIPELINE")
    print(f"{'='*70}\n")
    
    # Create pipeline
    detection_pipeline = DetectionPipeline(config)
    
    # Process video
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {config.video_path}")
        return
    
    print(f"✅ Video opened: {config.video_path}")
    print(f"Access preview at: http://localhost:5000\n")
    
    frame_num = 0
    inference_times = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"\n🔄 Looping video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue
            
            # Process frame through pipeline
            result = detection_pipeline.process_frame(frame, frame_num)
            
            if result:
                detections = result['detections']
                fraud = result['fraud']
                
                # Track inference time
                inf_time = detections.get('inference_time', 0)
                inference_times.append(inf_time)
                if len(inference_times) > 30:
                    inference_times.pop(0)
                avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
                fps = 1000 / avg_inference if avg_inference > 0 else 0
                
                # Draw frame with overlays
                display_frame = cv2.resize(frame, (1280, 720))
                display_frame = _draw_detections(display_frame, detections)
                
                # Update display (pass frame_num for face tracking)
                update_frame(display_frame, frame_num=frame_num)
                
                # Update stats
                stats.update({
                    'model': config.detection.model_name.upper(),
                    'fps': fps,
                    'detections': len(detections.get('boxes', [])),
                    'frame_num': frame_num,
                    'total_frames': config.metadata.get('total_frames', 0),
                    'fraud_events': detection_pipeline.stats['fraud_events'],
                    'inference_time_ms': avg_inference,
                    'pos_matches': detection_pipeline.stats['pos_matches'],
                    'pos_mismatches': detection_pipeline.stats['pos_mismatches']
                })
                
                # Progress indicator
                if frame_num % 30 == 0:
                    print(f"  Frame {frame_num} | Detections: {len(detections.get('boxes', []))} | "
                          f"Frauds: {detection_pipeline.stats['fraud_events']} | FPS: {fps:.1f}")
            
            frame_num += 1
    
    except Exception as e:
        print(f"ERROR in detection loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        detection_pipeline._finalize()

def _draw_detections(frame, detections):
    """Draw detection boxes on frame"""
    frame = frame.copy()
    
    boxes = detections.get('boxes', [])
    labels = detections.get('labels', [])
    scores = detections.get('scores', [])
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        label = labels[i] if i < len(labels) else 'object'
        score = scores[i] if i < len(scores) else 0
        
        color = (0, int(255 * score), int(255 * (1 - score)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Stats overlay
    cv2.putText(frame, f"Detections: {len(boxes)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {stats['inference_time_ms']:.1f}ms", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def run_server(config_path: str = None, port: int = 5000):
    """Start the Flask preview server"""
    global face_masker, detection_thread
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = load_config_from_file(config_path)
    else:
        print("Creating default configuration...")
        config = ApplicationConfig(
            video_path='videos/NVR_ch10_main_20260109095150_20260109095555.mp4',
            detection=DetectionConfig(model_name='yoloworld'),
            optimization=OptimizationConfig(model_name='yoloworld', skip_every_n_frames=1),
            pos=POSConfig(enabled=True),
            evidence=EvidenceConfig(enabled=True)
        )
        config.validate()
    
    # Initialize face masker with config (async MediaPipe by default)
    print("\n🔒 Initializing face masking for privacy protection...")
    
    # Get face masking config from ApplicationConfig if available
    face_masking_config = getattr(config, 'face_masking', None)
    if face_masking_config is not None:
        # Convert config_models.FaceMaskingConfig to face_masking.FaceMaskingConfig
        fm_config = FMConfig(
            enabled=face_masking_config.enabled,
            async_enabled=face_masking_config.async_enabled,
            detector_type=face_masking_config.detector_type,
            mask_type=face_masking_config.mask_type,
            blur_strength=face_masking_config.blur_strength,
            min_detection_confidence=face_masking_config.min_detection_confidence,
            persistence_frames=face_masking_config.persistence_frames,
            detection_interval_frames=face_masking_config.detection_interval_frames,
            enable_profile_detection=face_masking_config.enable_profile_detection,
            model_selection=face_masking_config.model_selection
        )
        reset_face_masker()  # Clear any existing instance
        face_masker = get_face_masker(config=fm_config)
    else:
        # Use default async MediaPipe config
        fm_config = FMConfig(
            async_enabled=True,
            detector_type="mediapipe",
            persistence_frames=15,
            detection_interval_frames=3
        )
        reset_face_masker()
        face_masker = get_face_masker(config=fm_config)
    
    # Store metadata
    cap = cv2.VideoCapture(config.video_path)
    if cap.isOpened():
        config.metadata['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    # Start detection thread
    detection_thread = threading.Thread(
        target=run_detection_loop,
        args=(config,),
        daemon=True
    )
    detection_thread.start()
    
    print(f"\n{'='*70}")
    print(f"🚀 Flask Preview Server Starting")
    print(f"🔒 Face masking: ENABLED (privacy mode)")
    print(f"{'='*70}")
    print(f"📺 Open in browser: http://localhost:{port}")
    print(f"{'='*70}\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flask Preview Server - Unified Detection Framework"
    )
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument("--config", default=None, help="Path to config file (YAML/JSON)")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--model", default="yoloworld", help="Detection model")
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    run_server(args.config, args.port)
