"""
Unified Configuration Models for Detection Pipeline
Handles all configuration for models, optimization, POS, evidence, and zones
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import json
import yaml
import os


class ModelType(Enum):
    """Model architecture types"""
    CNN = "cnn"
    VIT = "vit"
    TRANSFORMER = "transformer"


class DataFormat(Enum):
    """POS data format types"""
    XML = "xml"
    CSV = "csv"
    API = "api"


class MatchStrategy(Enum):
    """POS item matching strategies"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SUBSTRING = "substring"


class RecordMode(Enum):
    """Evidence recording modes"""
    ALL_FRAMES = "all_frames"
    FRAUD_ONLY = "fraud_only"
    FRAUD_WITH_BUFFER = "fraud_with_buffer"


class FaceDetectorType(Enum):
    """Face detector types"""
    MEDIAPIPE = "mediapipe"
    HAAR = "haar"
    DNN = "dnn"


class MaskType(Enum):
    """Face mask types"""
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK = "black"
    EMOJI = "emoji"


@dataclass
class FaceMaskingConfig:
    """Face masking settings for privacy protection"""
    enabled: bool = True
    async_enabled: bool = True
    detector_type: str = "mediapipe"  # mediapipe, haar, dnn
    mask_type: str = "blur"  # blur, pixelate, black, emoji
    blur_strength: int = 51
    min_detection_confidence: float = 0.5
    persistence_frames: int = 15  # Continue blur for N frames after face lost
    detection_interval_frames: int = 2  # Detect every N frames, interpolate between
    enable_profile_detection: bool = True  # Use profile face cascade in addition to frontal
    model_selection: int = 0  # MediaPipe Tasks: 0=short-range (available)


@dataclass
class DetectionConfig:
    """Per-model detection settings"""
    model_name: str  # 'yoloworld', 'owlv2', 'yoloe', 'groundingdino'
    model_type: ModelType = ModelType.CNN
    model_path: Optional[str] = None
    weights_path: Optional[str] = None
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.5
    device: str = 'cuda'
    enable_fp16: bool = False
    
    def __post_init__(self):
        """Validate model_name"""
        valid_models = ['yoloworld', 'owlv2', 'yoloe', 'groundingdino']
        if self.model_name.lower() not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}, got {self.model_name}")


@dataclass
class OptimizationConfig:
    """Performance tuning per model"""
    model_name: str
    enable_roi_crop: bool = False
    roi_bounds: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    input_width: int = 1280
    input_height: int = 720
    max_dim: int = 1280  # max(width, height) after resize
    skip_every_n_frames: int = 1  # Detect every Nth frame (1 = every frame)
    target_fps: int = 30  # Target FPS (auto-calculate skip if needed)
    clear_gpu_after_n_frames: int = 30  # Memory management
    enable_batch_processing: bool = False
    batch_size: int = 1


@dataclass
class POSConfig:
    """POS data integration settings"""
    enabled: bool = True
    xml_path: Optional[str] = None
    data_format: DataFormat = DataFormat.XML
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timezone_offset: int = 0  # Handle time zone differences in hours
    match_strategy: MatchStrategy = MatchStrategy.FUZZY
    min_match_confidence: float = 0.7  # For fuzzy matching
    item_name_field: str = "item"  # Field name for item description
    timestamp_field: str = "time"  # Field name for scan timestamp


@dataclass
class EvidenceConfig:
    """Evidence recording settings"""
    enabled: bool = True
    output_dir: str = 'evidence'
    record_mode: RecordMode = RecordMode.FRAUD_WITH_BUFFER
    buffer_seconds_before: float = 3.0
    buffer_seconds_after: float = 5.0
    save_frames: bool = True
    save_video_clips: bool = True
    save_fraud_report: bool = True
    image_quality: int = 95  # JPEG quality (0-100)
    video_codec: str = 'avc1'  # Video codec (avc1/H264 recommended for browser playback)
    enable_face_masking: bool = True  # Blur faces in evidence
    frame_text_overlay: bool = True  # Show frame number + timestamp


@dataclass
class ZoneConfig:
    """Zone definitions for fraud detection"""
    counter: List[Tuple[int, int]] = field(default_factory=list)  # Polygon points (x, y)
    scanner: List[Tuple[int, int]] = field(default_factory=list)
    trolley: List[Tuple[int, int]] = field(default_factory=list)
    exit: List[Tuple[int, int]] = field(default_factory=list)
    pos: List[Tuple[int, int]] = field(default_factory=list)
    customer_area: List[Tuple[int, int]] = field(default_factory=list)
    baby_seat: List[Tuple[int, int]] = field(default_factory=list)
    basket: List[Tuple[int, int]] = field(default_factory=list)
    cashier: List[Tuple[int, int]] = field(default_factory=list)
    packing_area: List[Tuple[int, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[Tuple[int, int]]]:
        """Convert to dict representation"""
        return {
            'counter': self.counter,
            'scanner': self.scanner,
            'trolley': self.trolley,
            'exit': self.exit,
            'pos': self.pos,
            'customer_area': self.customer_area,
            'baby_seat': self.baby_seat,
            'basket': self.basket,
            'cashier': self.cashier,
            'packing_area': self.packing_area
        }


@dataclass
class DetectionClassesConfig:
    """Configuration for detection class names"""
    retail_classes: List[str] = field(default_factory=lambda: [
        "bottle", "box", "bag", "item", "product"
    ])
    person_classes: List[str] = field(default_factory=lambda: [
        "person"
    ])
    scanner_classes: List[str] = field(default_factory=lambda: [
        "scanner"
    ])


@dataclass
class ApplicationConfig:
    """Top-level application configuration"""
    # Video settings
    video_path: str
    fps: int = 25
    frame_width: int = 1280
    frame_height: int = 720
    
    # Detection
    detection: DetectionConfig = field(default_factory=lambda: DetectionConfig(model_name='yoloworld'))
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(model_name='yoloworld'))
    
    # POS integration
    pos: POSConfig = field(default_factory=POSConfig)
    
    # Evidence recording
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    
    # Face masking for privacy
    face_masking: FaceMaskingConfig = field(default_factory=FaceMaskingConfig)
    
    # Zones
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    
    # Classes
    classes: DetectionClassesConfig = field(default_factory=DetectionClassesConfig)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        if not self.video_path:
            raise ValueError("video_path is required")
        
        if self.detection.model_name != self.optimization.model_name:
            raise ValueError(
                f"detection.model_name ({self.detection.model_name}) must match "
                f"optimization.model_name ({self.optimization.model_name})"
            )
        
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ApplicationConfig':
        """Create config from dictionary (e.g., from YAML/JSON)"""
        # Parse enums
        if 'pos' in config_dict and isinstance(config_dict['pos'], dict):
            if 'data_format' in config_dict['pos']:
                config_dict['pos']['data_format'] = DataFormat(config_dict['pos']['data_format'])
            if 'match_strategy' in config_dict['pos']:
                config_dict['pos']['match_strategy'] = MatchStrategy(config_dict['pos']['match_strategy'])
            config_dict['pos'] = POSConfig(**config_dict['pos'])
        
        if 'evidence' in config_dict and isinstance(config_dict['evidence'], dict):
            if 'record_mode' in config_dict['evidence']:
                config_dict['evidence']['record_mode'] = RecordMode(config_dict['evidence']['record_mode'])
            config_dict['evidence'] = EvidenceConfig(**config_dict['evidence'])
        
        if 'detection' in config_dict and isinstance(config_dict['detection'], dict):
            if 'model_type' in config_dict['detection']:
                config_dict['detection']['model_type'] = ModelType(config_dict['detection']['model_type'])
            config_dict['detection'] = DetectionConfig(**config_dict['detection'])
        
        if 'optimization' in config_dict and isinstance(config_dict['optimization'], dict):
            config_dict['optimization'] = OptimizationConfig(**config_dict['optimization'])
        
        if 'face_masking' in config_dict and isinstance(config_dict['face_masking'], dict):
            config_dict['face_masking'] = FaceMaskingConfig(**config_dict['face_masking'])
        
        if 'zones' in config_dict and isinstance(config_dict['zones'], dict):
            config_dict['zones'] = ZoneConfig(**config_dict['zones'])
        
        if 'classes' in config_dict and isinstance(config_dict['classes'], dict):
            config_dict['classes'] = DetectionClassesConfig(**config_dict['classes'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = asdict(self)
        
        # Helper function to convert enum to string value
        def enum_to_value(val):
            """Convert enum to its value string, or return as-is if already string"""
            if isinstance(val, Enum):
                return val.value
            return val
        
        # Convert enums to their values
        if 'pos' in config_dict and isinstance(config_dict['pos'], dict):
            if 'data_format' in config_dict['pos']:
                config_dict['pos']['data_format'] = enum_to_value(config_dict['pos']['data_format'])
            if 'match_strategy' in config_dict['pos']:
                config_dict['pos']['match_strategy'] = enum_to_value(config_dict['pos']['match_strategy'])
        
        if 'evidence' in config_dict and isinstance(config_dict['evidence'], dict):
            if 'record_mode' in config_dict['evidence']:
                config_dict['evidence']['record_mode'] = enum_to_value(config_dict['evidence']['record_mode'])
        
        if 'detection' in config_dict and isinstance(config_dict['detection'], dict):
            if 'model_type' in config_dict['detection']:
                config_dict['detection']['model_type'] = enum_to_value(config_dict['detection']['model_type'])
        
        return config_dict
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to YAML or JSON file"""
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        config_dict = self.to_dict()
        
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            # Default to YAML
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ApplicationConfig':
        """Load configuration from YAML or JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config_dict = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                # Try YAML first, then JSON
                try:
                    config_dict = yaml.safe_load(f)
                except:
                    f.seek(0)
                    config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Preset configurations for quick use
PRESET_CONFIGS = {
    'fast_yoloworld': {
        'detection': DetectionConfig(
            model_name='yoloworld',
            model_type=ModelType.CNN,
            confidence_threshold=0.20,
            iou_threshold=0.45
        ),
        'optimization': OptimizationConfig(
            model_name='yoloworld',
            max_dim=640,
            skip_every_n_frames=1
        )
    },
    'accurate_owlv2': {
        'detection': DetectionConfig(
            model_name='owlv2',
            model_type=ModelType.VIT,
            confidence_threshold=0.15,
            iou_threshold=0.5
        ),
        'optimization': OptimizationConfig(
            model_name='owlv2',
            max_dim=960,
            enable_roi_crop=True,
            skip_every_n_frames=2
        )
    },
    'balanced_yoloe': {
        'detection': DetectionConfig(
            model_name='yoloe',
            model_type=ModelType.CNN,
            confidence_threshold=0.15,
            iou_threshold=0.5
        ),
        'optimization': OptimizationConfig(
            model_name='yoloe',
            max_dim=800,
            skip_every_n_frames=1
        )
    }
}
