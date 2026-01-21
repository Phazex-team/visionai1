"""
Face Masking Module for Privacy Protection
Detects and masks all faces in frames before display/saving.
Supports async detection with MediaPipe for better accuracy and temporal persistence.
"""
import cv2
import numpy as np
import os
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

# Try to import MediaPipe, fall back gracefully if not available
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_SOLUTIONS_AVAILABLE = False
MEDIAPIPE_TASKS_AVAILABLE = False

mp = None
mp_vision = None
BaseOptions = None

try:
    import mediapipe as mp  # type: ignore

    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions")

    try:
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore

        MEDIAPIPE_TASKS_AVAILABLE = True
    except Exception:
        MEDIAPIPE_TASKS_AVAILABLE = False

except ImportError:
    mp = None
    print("[FACE MASKING] WARNING: MediaPipe not installed. Using Haar cascade fallback.")
    print("[FACE MASKING] Install with: pip install mediapipe")


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEDIAPIPE_MODEL_DIR = _REPO_ROOT / "weights" / "mediapipe"


class DetectorType(Enum):
    """Face detector types"""
    MEDIAPIPE = "mediapipe"
    HAAR = "haar"
    DNN = "dnn"


@dataclass
class FaceMaskingConfig:
    """Configuration for face masking"""
    enabled: bool = True
    async_enabled: bool = True
    detector_type: str = "mediapipe"  # mediapipe, haar, dnn
    mask_type: str = "blur"  # blur, pixelate, black, emoji
    blur_strength: int = 51
    min_detection_confidence: float = 0.5
    persistence_frames: int = 15  # Continue blur for N frames after face lost
    detection_interval_frames: int = 3  # Detect every N frames, interpolate between
    enable_profile_detection: bool = True  # Use profile face cascade in addition to frontal
    model_selection: int = 0  # MediaPipe Tasks: 0=short-range (available). Full-range model is not currently published.
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FaceMaskingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'async_enabled': self.async_enabled,
            'detector_type': self.detector_type,
            'mask_type': self.mask_type,
            'blur_strength': self.blur_strength,
            'min_detection_confidence': self.min_detection_confidence,
            'persistence_frames': self.persistence_frames,
            'detection_interval_frames': self.detection_interval_frames,
            'enable_profile_detection': self.enable_profile_detection,
            'model_selection': self.model_selection
        }


@dataclass
class TrackedFace:
    """Represents a tracked face with temporal persistence"""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    last_seen_frame: int
    confidence: float
    velocity: Tuple[float, float] = (0.0, 0.0)  # For position prediction
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def update(self, bbox: Tuple[int, int, int, int], frame_num: int, confidence: float):
        """Update face position and calculate velocity"""
        old_center = (self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2)
        new_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        
        # Calculate velocity for prediction
        frame_diff = max(1, frame_num - self.last_seen_frame)
        self.velocity = (
            (new_center[0] - old_center[0]) / frame_diff,
            (new_center[1] - old_center[1]) / frame_diff
        )
        
        self.bbox = bbox
        self.last_seen_frame = frame_num
        self.confidence = confidence
        self.history.append(bbox)
    
    def predict_position(self, current_frame: int) -> Tuple[int, int, int, int]:
        """Predict face position based on velocity"""
        frames_since_seen = current_frame - self.last_seen_frame
        if frames_since_seen == 0:
            return self.bbox
        
        # Predict new center based on velocity
        x, y, w, h = self.bbox
        pred_x = int(x + self.velocity[0] * frames_since_seen)
        pred_y = int(y + self.velocity[1] * frames_since_seen)
        
        return (pred_x, pred_y, w, h)


class AsyncFaceDetector:
    """
    Asynchronous face detection running on a separate thread.
    Provides non-blocking detection with result caching.
    """
    
    def __init__(self, config: FaceMaskingConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=2)
        self._result_lock = threading.Lock()
        self._latest_faces: List[Tuple[int, int, int, int]] = []
        self._latest_frame_num = -1
        self._detector_thread: Optional[threading.Thread] = None
        self._detector = None
        self._initialized = False
        
        # MediaPipe face detection
        self._mp_face_detection = None
        self._mp_task_detector = None
        
        # Haar cascades for fallback
        self._face_cascade = None
        self._profile_cascade = None
        
        # DNN model
        self._dnn_net = None
        
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the face detector based on config"""
        detector_type = self.config.detector_type.lower()
        
        # Try MediaPipe first if requested
        if detector_type == "mediapipe" and MEDIAPIPE_AVAILABLE:
            # Older API: mp.solutions (not available in some builds, e.g. Py3.12 wheels)
            if MEDIAPIPE_SOLUTIONS_AVAILABLE:
                try:
                    self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
                        model_selection=self.config.model_selection,
                        min_detection_confidence=self.config.min_detection_confidence
                    )
                    self._detector = "mediapipe_solutions"
                    self._initialized = True
                    print(f"[FACE MASKING] Using MediaPipe Solutions FaceDetection (model={self.config.model_selection})")
                    return
                except Exception as e:
                    print(f"[FACE MASKING] MediaPipe Solutions init failed: {e}, falling back")

            # Newer API: MediaPipe Tasks
            if MEDIAPIPE_TASKS_AVAILABLE and mp_vision is not None and BaseOptions is not None:
                try:
                    model_path = self._get_mediapipe_tasks_model_path(self.config.model_selection)
                    options = mp_vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=str(model_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        min_detection_confidence=float(self.config.min_detection_confidence),
                    )
                    self._mp_task_detector = mp_vision.FaceDetector.create_from_options(options)
                    self._detector = "mediapipe_tasks"
                    self._initialized = True
                    print(f"[FACE MASKING] Using MediaPipe Tasks FaceDetector (model={model_path.name})")
                    return
                except Exception as e:
                    print(f"[FACE MASKING] MediaPipe Tasks init failed: {e}, falling back to Haar")
        
        # Try DNN if requested
        if detector_type == "dnn":
            dnn_paths = [
                ('/usr/share/opencv4/dnn/deploy.prototxt', 
                 '/usr/share/opencv4/dnn/res10_300x300_ssd_iter_140000.caffemodel'),
                ('/usr/local/share/opencv4/dnn/deploy.prototxt', 
                 '/usr/local/share/opencv4/dnn/res10_300x300_ssd_iter_140000.caffemodel'),
            ]
            for prototxt, model in dnn_paths:
                if os.path.exists(prototxt) and os.path.exists(model):
                    try:
                        self._dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
                        self._detector = "dnn"
                        self._initialized = True
                        print("[FACE MASKING] Using OpenCV DNN face detector")
                        return
                    except Exception as e:
                        print(f"[FACE MASKING] DNN init failed: {e}")
        
        # Fall back to Haar cascades
        self._init_haar_cascades()

    def _get_mediapipe_tasks_model_path(self, model_selection: int) -> Path:
        """Resolve local TFLite model path for MediaPipe Tasks FaceDetector."""
        # model_selection: 0=short range, 1=full range
        short_name = "blaze_face_short_range.tflite"
        full_name = "blaze_face_full_range.tflite"

        if int(model_selection) == 0:
            model_path = _MEDIAPIPE_MODEL_DIR / short_name
            if model_path.exists():
                return model_path
            raise FileNotFoundError(
                f"MediaPipe Tasks model not found: {model_path}. Expected under {_MEDIAPIPE_MODEL_DIR}."
            )

        # Full-range BlazeFace is currently not published for Tasks (the official docs list it as coming soon).
        # Fall back to short-range model to keep MediaPipe enabled.
        fallback_path = _MEDIAPIPE_MODEL_DIR / short_name
        if fallback_path.exists():
            print("[FACE MASKING] WARNING: Requested full-range model, falling back to short-range BlazeFace")
            return fallback_path

        raise FileNotFoundError(
            f"MediaPipe Tasks model not found: {fallback_path}. Expected under {_MEDIAPIPE_MODEL_DIR}."
        )
    
    def _init_haar_cascades(self):
        """Initialize Haar cascade detectors"""
        # Frontal face cascade
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                self._face_cascade = cv2.CascadeClassifier(path)
                if not self._face_cascade.empty():
                    print(f"[FACE MASKING] Using Haar frontal cascade: {os.path.basename(path)}")
                    break
        
        if self._face_cascade is None or self._face_cascade.empty():
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # Profile face cascade (if enabled)
        if self.config.enable_profile_detection:
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            if os.path.exists(profile_path):
                self._profile_cascade = cv2.CascadeClassifier(profile_path)
                if not self._profile_cascade.empty():
                    print("[FACE MASKING] Profile face detection ENABLED")
        
        self._detector = "haar"
        self._initialized = True
    
    def start(self):
        """Start the async detection thread"""
        if self._detector_thread is not None and self._detector_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._detector_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="FaceDetectorThread"
        )
        self._detector_thread.start()
        print("[FACE MASKING] Async detection thread started")
    
    def stop(self):
        """Stop the async detection thread"""
        self._stop_event.set()
        if self._detector_thread is not None:
            self._detector_thread.join(timeout=2.0)
            self._detector_thread = None
        
        # Release MediaPipe resources
        if self._mp_face_detection is not None:
            self._mp_face_detection.close()
            self._mp_face_detection = None

        if self._mp_task_detector is not None:
            try:
                self._mp_task_detector.close()
            except Exception:
                pass
            self._mp_task_detector = None
        
        print("[FACE MASKING] Async detection thread stopped")
    
    def submit_frame(self, frame: np.ndarray, frame_num: int):
        """Submit a frame for async detection (non-blocking)"""
        try:
            # Drop old frames if queue is full
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self._frame_queue.put_nowait((frame.copy(), frame_num))
        except queue.Full:
            pass  # Skip if queue is full
    
    def get_latest_faces(self) -> Tuple[List[Tuple[int, int, int, int]], int]:
        """Get the most recent detection results"""
        with self._result_lock:
            return self._latest_faces.copy(), self._latest_frame_num
    
    def _detection_loop(self):
        """Main detection loop running in background thread"""
        while not self._stop_event.is_set():
            try:
                # Wait for frame with timeout
                frame, frame_num = self._frame_queue.get(timeout=0.1)
                
                # Perform detection (with fallback if primary returns nothing)
                faces = self._detect_faces_with_fallback(frame)
                
                # Update results
                with self._result_lock:
                    self._latest_faces = faces
                    self._latest_frame_num = frame_num
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FACE MASKING] Detection error: {e}")
                continue
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using the configured detector"""
        if frame is None:
            return []
        
        faces = []
        
        if self._detector == "mediapipe_solutions" and self._mp_face_detection is not None:
            faces = self._detect_mediapipe(frame)
        elif self._detector == "mediapipe_tasks" and self._mp_task_detector is not None:
            faces = self._detect_mediapipe_tasks(frame)
        elif self._detector == "dnn" and self._dnn_net is not None:
            faces = self._detect_dnn(frame)
        else:
            faces = self._detect_haar(frame)
        
        return faces

    def _detect_faces_with_fallback(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces with fallback to Haar if primary detector finds nothing.
        This is useful because MediaPipe BlazeFace is optimized for close-up selfie/webcam
        scenarios and may miss small/distant faces in surveillance footage.
        """
        faces = self._detect_faces(frame)
        
        # If primary detector (MediaPipe) found no faces, try Haar as fallback
        if not faces and self._detector in ("mediapipe_solutions", "mediapipe_tasks", "dnn"):
            # Ensure Haar cascades are initialized
            if self._face_cascade is None:
                self._init_haar_cascades()
            faces = self._detect_haar(frame)
        
        return faces
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        faces = []
        h, w = frame.shape[:2]
        
        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coords to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                face_w = min(face_w, w - x)
                face_h = min(face_h, h - y)
                
                if face_w > 10 and face_h > 10:
                    faces.append((x, y, face_w, face_h))
        
        return faces

    def _detect_mediapipe_tasks(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe Tasks FaceDetector."""
        faces: List[Tuple[int, int, int, int]] = []
        if self._mp_task_detector is None or mp is None:
            return faces

        h, w = frame.shape[:2]

        # Tasks API expects SRGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._mp_task_detector.detect(mp_image)
        if result is None or not getattr(result, "detections", None):
            return faces

        for det in result.detections:
            bbox = getattr(det, "bounding_box", None)
            if bbox is None:
                continue

            x = int(getattr(bbox, "origin_x", 0))
            y = int(getattr(bbox, "origin_y", 0))
            face_w = int(getattr(bbox, "width", 0))
            face_h = int(getattr(bbox, "height", 0))

            # Clamp
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            face_w = max(0, min(face_w, w - x))
            face_h = max(0, min(face_h, h - y))

            if face_w > 10 and face_h > 10:
                faces.append((x, y, face_w, face_h))

        return faces
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN"""
        faces = []
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), 
            (104.0, 177.0, 123.0)
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config.min_detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascades"""
        faces = []
        h, w = frame.shape[:2]
        
        # For high resolution images, scale down for faster detection
        max_detection_size = 1280
        scale = 1.0
        if max(w, h) > max_detection_size:
            scale = max_detection_size / max(w, h)
            detection_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            detection_frame = frame
        
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        
        # Frontal face detection
        if self._face_cascade is not None and not self._face_cascade.empty():
            detected = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Scale coordinates back to original size
            for (x, y, fw, fh) in detected:
                faces.append((int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)))
        
        # Profile face detection (left and right)
        if self._profile_cascade is not None and not self._profile_cascade.empty():
            # Left profile
            profile_left = self._profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, fw, fh) in profile_left:
                faces.append((int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)))
            
            # Right profile (flip image)
            gray_flipped = cv2.flip(gray, 1)
            profile_right = self._profile_cascade.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            # Flip coordinates back
            h_det, w_det = gray.shape[:2]
            for (x, y, fw, fh) in profile_right:
                orig_x = int((w_det - x - fw) / scale)
                faces.append((orig_x, int(y/scale), int(fw/scale), int(fh/scale)))
        
        return faces
    
    def detect_sync(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Synchronous detection (blocking) - for single frame use, with fallback"""
        return self._detect_faces_with_fallback(frame)


class FaceMasker:
    """
    Face detection and masking for privacy protection with temporal persistence.
    Supports async detection with MediaPipe and face tracking across frames.
    Does NOT affect the original frame used for detection/tracking.
    """
    
    def __init__(
        self, 
        mask_type: str = "blur", 
        blur_strength: int = 51,
        config: Optional[FaceMaskingConfig] = None
    ):
        """
        Initialize the face masker.
        
        Args:
            mask_type: "blur" for gaussian blur, "black" for black boxes, "pixelate" for pixelation
            blur_strength: Blur kernel size (must be odd number)
            config: Optional FaceMaskingConfig for advanced settings
        """
        # Create config from params if not provided
        if config is None:
            config = FaceMaskingConfig(
                mask_type=mask_type,
                blur_strength=blur_strength
            )
        
        self.config = config
        self.mask_type = config.mask_type
        self.blur_strength = config.blur_strength if config.blur_strength % 2 == 1 else config.blur_strength + 1
        
        # Face tracking state
        self._tracked_faces: Dict[int, TrackedFace] = {}
        self._next_face_id = 0
        self._current_frame_num = 0
        self._track_lock = threading.Lock()
        
        # Cached face positions for fast masking
        self._cached_faces: List[Tuple[int, int, int, int]] = []
        self._cache_frame_num = -1
        
        # Initialize async detector
        self._async_detector: Optional[AsyncFaceDetector] = None
        if config.async_enabled:
            self._async_detector = AsyncFaceDetector(config)
            self._async_detector.start()
        else:
            # Sync-only detector for fallback
            self._sync_detector = AsyncFaceDetector(config)
        
        # Legacy compatibility
        self.face_cascade = None
        self.dnn_net = None
        self.use_dnn = False
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)
        self.confidence_threshold = config.min_detection_confidence
        
        print(f"[FACE MASKING] Initialized with {self.mask_type} mask, "
              f"async={'ENABLED' if config.async_enabled else 'DISABLED'}, "
              f"detector={config.detector_type}")
    
    def stop(self):
        """Stop async detection and release resources"""
        if self._async_detector is not None:
            self._async_detector.stop()
            self._async_detector = None
        print("[FACE MASKING] Face masker stopped")
    
    def _match_face_to_tracked(
        self, 
        bbox: Tuple[int, int, int, int],
        confidence: float = 0.5
    ) -> Optional[int]:
        """
        Match a detected face to an existing tracked face.
        Returns face_id if matched, None if new face.
        """
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)
        
        best_match_id = None
        best_distance = float('inf')
        max_match_distance = max(w, h) * 1.5  # Allow movement up to 1.5x face size
        
        for face_id, tracked in self._tracked_faces.items():
            tx, ty, tw, th = tracked.bbox
            tracked_center = (tx + tw // 2, ty + th // 2)
            
            # Calculate distance between centers
            distance = np.sqrt((center[0] - tracked_center[0])**2 + 
                              (center[1] - tracked_center[1])**2)
            
            if distance < max_match_distance and distance < best_distance:
                best_distance = distance
                best_match_id = face_id
        
        return best_match_id
    
    def _update_tracking(
        self, 
        detected_faces: List[Tuple[int, int, int, int]], 
        frame_num: int
    ):
        """Update face tracking with new detections"""
        with self._track_lock:
            self._current_frame_num = frame_num
            matched_ids = set()
            
            # Match detections to tracked faces
            for bbox in detected_faces:
                face_id = self._match_face_to_tracked(bbox)
                
                if face_id is not None:
                    # Update existing tracked face
                    self._tracked_faces[face_id].update(bbox, frame_num, 0.9)
                    matched_ids.add(face_id)
                else:
                    # Create new tracked face
                    new_face = TrackedFace(
                        face_id=self._next_face_id,
                        bbox=bbox,
                        last_seen_frame=frame_num,
                        confidence=0.9
                    )
                    new_face.history.append(bbox)
                    self._tracked_faces[self._next_face_id] = new_face
                    self._next_face_id += 1
            
            # Remove stale tracked faces (not seen for persistence_frames)
            stale_ids = []
            for face_id, tracked in self._tracked_faces.items():
                frames_since_seen = frame_num - tracked.last_seen_frame
                if frames_since_seen > self.config.persistence_frames:
                    stale_ids.append(face_id)
            
            for face_id in stale_ids:
                del self._tracked_faces[face_id]
    
    def _get_faces_to_mask(self, frame_num: int) -> List[Tuple[int, int, int, int]]:
        """
        Get all face bounding boxes to mask, including predicted positions
        for faces not seen in current frame (persistence).
        """
        faces_to_mask = []
        
        with self._track_lock:
            for face_id, tracked in self._tracked_faces.items():
                frames_since_seen = frame_num - tracked.last_seen_frame
                
                if frames_since_seen == 0:
                    # Face was detected this frame
                    faces_to_mask.append(tracked.bbox)
                elif frames_since_seen <= self.config.persistence_frames:
                    # Face not detected but within persistence window - use predicted position
                    predicted_bbox = tracked.predict_position(frame_num)
                    faces_to_mask.append(predicted_bbox)
        
        return faces_to_mask
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame (sync mode for compatibility).
        
        Args:
            frame: BGR image
            
        Returns:
            List of (x, y, w, h) tuples for each detected face
        """
        if frame is None:
            return []
        
        if self._async_detector is not None:
            return self._async_detector.detect_sync(frame)
        elif hasattr(self, '_sync_detector'):
            return self._sync_detector.detect_sync(frame)
        else:
            return []
    
    def mask_face(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Apply mask to a single face region.
        
        Args:
            frame: Image to modify (will be modified in place)
            x, y, w, h: Face bounding box
            
        Returns:
            Modified frame
        """
        # Add padding around face
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        if self.mask_type == "blur":
            # Gaussian blur
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                blurred = cv2.GaussianBlur(face_region, (self.blur_strength, self.blur_strength), 0)
                frame[y1:y2, x1:x2] = blurred
                
        elif self.mask_type == "pixelate":
            # Pixelation effect
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                # Shrink and expand
                small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = pixelated
                
        elif self.mask_type == "black":
            # Solid black box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
        elif self.mask_type == "emoji":
            # Could add emoji overlay here
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            # Draw a simple smiley
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = min(w, h) // 2
            cv2.circle(frame, (cx, cy), radius, (0, 255, 255), -1)
            # Eyes
            cv2.circle(frame, (cx - radius//3, cy - radius//4), radius//8, (0, 0, 0), -1)
            cv2.circle(frame, (cx + radius//3, cy - radius//4), radius//8, (0, 0, 0), -1)
            # Smile
            cv2.ellipse(frame, (cx, cy + radius//4), (radius//3, radius//4), 0, 0, 180, (0, 0, 0), 2)
        
        return frame
    
    def mask_faces(
        self, 
        frame: np.ndarray, 
        faces: List[Tuple[int, int, int, int]] = None,
        frame_num: int = None
    ) -> np.ndarray:
        """
        Mask all faces in frame with temporal persistence.
        
        Args:
            frame: Original frame (will NOT be modified)
            faces: Optional pre-detected faces. If None, will use async detection.
            frame_num: Frame number for tracking (auto-increments if not provided)
            
        Returns:
            New frame with faces masked
        """
        if frame is None:
            return frame
        
        # Auto-increment frame number if not provided
        if frame_num is None:
            frame_num = self._current_frame_num + 1
        
        self._current_frame_num = frame_num
        
        # Always work on a copy to preserve original for detection
        masked_frame = frame.copy()
        
        # Get detected faces from async detector or use provided
        detected_faces = faces
        if detected_faces is None:
            if self.config.async_enabled and self._async_detector is not None:
                # Check if we should run detection this frame
                if frame_num % self.config.detection_interval_frames == 0:
                    self._async_detector.submit_frame(frame, frame_num)
                
                # Get latest detection results
                detected_faces, detect_frame = self._async_detector.get_latest_faces()
                
                # If we have recent detections, update tracking
                if detect_frame >= 0 and detected_faces:
                    self._update_tracking(detected_faces, detect_frame)
            else:
                # Sync detection
                if frame_num % self.config.detection_interval_frames == 0 or not self._tracked_faces:
                    detected_faces = self.detect_faces(frame)
                    self._update_tracking(detected_faces, frame_num)
                else:
                    detected_faces = []
        else:
            # External faces provided - update tracking
            self._update_tracking(detected_faces, frame_num)
        
        # Get all faces to mask (including persistent/predicted positions)
        faces_to_mask = self._get_faces_to_mask(frame_num)
        
        # Mask each face
        for (x, y, w, h) in faces_to_mask:
            self.mask_face(masked_frame, x, y, w, h)
        
        return masked_frame
    
    def process_frame(self, frame: np.ndarray, frame_num: int = None) -> Tuple[np.ndarray, int]:
        """
        Process a frame: detect and mask all faces.
        
        Args:
            frame: Original frame
            frame_num: Frame number for tracking
            
        Returns:
            Tuple of (masked_frame, num_faces_masked)
        """
        if frame is None:
            return frame, 0
        
        masked_frame = self.mask_faces(frame, frame_num=frame_num)
        faces_to_mask = self._get_faces_to_mask(self._current_frame_num)
        
        return masked_frame, len(faces_to_mask)
    
    def get_tracked_faces_count(self) -> int:
        """Get the number of currently tracked faces"""
        with self._track_lock:
            return len(self._tracked_faces)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get face masking statistics"""
        with self._track_lock:
            return {
                'tracked_faces': len(self._tracked_faces),
                'current_frame': self._current_frame_num,
                'detector_type': self.config.detector_type,
                'async_enabled': self.config.async_enabled,
                'persistence_frames': self.config.persistence_frames
            }


# Global face masker instance
_face_masker = None
_face_masker_lock = threading.Lock()


def get_face_masker(
    mask_type: str = "blur", 
    blur_strength: int = 51,
    config: Optional[FaceMaskingConfig] = None
) -> FaceMasker:
    """
    Get or create the global face masker instance.
    
    Args:
        mask_type: Mask type (blur, pixelate, black, emoji)
        blur_strength: Gaussian blur kernel size
        config: Optional FaceMaskingConfig for advanced settings
    """
    global _face_masker
    with _face_masker_lock:
        if _face_masker is None:
            if config is None:
                config = FaceMaskingConfig(
                    mask_type=mask_type,
                    blur_strength=blur_strength
                )
            _face_masker = FaceMasker(config=config)
    return _face_masker


def reset_face_masker():
    """Reset the global face masker instance (for config changes)"""
    global _face_masker
    with _face_masker_lock:
        if _face_masker is not None:
            _face_masker.stop()
            _face_masker = None


def mask_faces_in_frame(frame: np.ndarray, frame_num: int = None) -> np.ndarray:
    """
    Convenience function to mask faces in a frame.
    
    Args:
        frame: Original frame (will NOT be modified)
        frame_num: Optional frame number for tracking
        
    Returns:
        New frame with faces masked
    """
    masker = get_face_masker()
    masked_frame, _ = masker.process_frame(frame, frame_num)
    return masked_frame


def create_face_masker_from_config(config_dict: Dict) -> FaceMasker:
    """
    Create a FaceMasker from a configuration dictionary.
    
    Args:
        config_dict: Dictionary with face masking configuration
        
    Returns:
        Configured FaceMasker instance
    """
    config = FaceMaskingConfig.from_dict(config_dict)
    return FaceMasker(config=config)
