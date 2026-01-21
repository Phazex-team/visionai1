"""
Abstract Model Interface for Fraud Detection
All models must implement this interface for consistent usage
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime

class DetectionModel(ABC):
    """Base class for all detection models"""
    
    def __init__(self, name: str, model_type: str, device: str = "cuda"):
        """
        Args:
            name: Model name (e.g., "GroundingDINO", "YOLOWorld")
            model_type: Type classification (e.g., "vit", "cnn", "transformer")
            device: "cuda" or "cpu"
        """
        self.name = name
        self.model_type = model_type
        self.device = device
        self.model = None
        
        # Configuration (set via set_detection_config)
        self.detection_config = None
        self.optimization_config = None
        
        # Metadata for calling code
        self.metadata = {}
        
        # Performance metrics
        self.metrics = {
            'inference_time': [],
            'detections_per_frame': [],
            'tracking_accuracy': 0,
            'fraud_detection_rate': 0,
            'false_positive_rate': 0,
            'total_frames_processed': 0
        }
    
    @abstractmethod
    def load_model(self):
        """Load and initialize the model"""
        pass
    
    @abstractmethod
    def set_classes(self, classes: List[str]):
        """Set detection classes for the model"""
        pass
    
    def set_detection_config(self, config: Any) -> None:
        """
        Set detection parameters from unified config
        
        Args:
            config: DetectionConfig object with thresholds and settings
        """
        self.detection_config = config
        # Subclasses should override to apply config-specific logic
        self._apply_detection_config()
    
    def _apply_detection_config(self) -> None:
        """
        Apply detection config settings to model.
        Override in subclasses to implement model-specific logic.
        Default: set confidence and iou thresholds if model supports it.
        """
        if not self.detection_config:
            return
        
        # Try to set thresholds (works for most models)
        try:
            self.set_thresholds(
                confidence=self.detection_config.confidence_threshold,
                iou=self.detection_config.iou_threshold
            )
        except (AttributeError, TypeError):
            # Model may not support set_thresholds, try set_threshold (OWLv2 legacy)
            try:
                self.set_threshold(self.detection_config.confidence_threshold)
            except (AttributeError, TypeError):
                # Model doesn't support threshold setting, skip silently
                pass
    
    def set_thresholds(self, confidence: float = 0.15, iou: float = 0.5) -> None:
        """
        Set confidence and IOU thresholds.
        Override in model-specific classes.
        
        Args:
            confidence: Confidence threshold (0-1)
            iou: IOU threshold for NMS (0-1)
        """
        pass
    
    def set_threshold(self, threshold: float) -> None:
        """
        Legacy: Set single threshold value.
        Override in model-specific classes if needed.
        """
        pass
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Optional per-model preprocessing (format conversion, normalization, etc.)
        Override in subclasses for model-specific preprocessing.
        
        Args:
            frame: Input frame (BGR from OpenCV)
            
        Returns:
            Preprocessed frame
        """
        return frame
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict:
        """
        Run inference on a frame
        
        Returns dict with:
        {
            'boxes': np.ndarray of shape (N, 4),  # [x1, y1, x2, y2]
            'scores': np.ndarray of shape (N,),   # confidence scores
            'labels': List[str],                   # class labels
            'inference_time': float                # milliseconds
        }
        """
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset any internal state (for batch processing)"""
        pass
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        return {
            'model_name': self.name,
            'model_type': self.model_type,
            'device': self.device,
            'avg_inference_time_ms': np.mean(self.metrics['inference_time']) if self.metrics['inference_time'] else 0,
            'avg_detections_per_frame': np.mean(self.metrics['detections_per_frame']) if self.metrics['detections_per_frame'] else 0,
            'tracking_accuracy': self.metrics['tracking_accuracy'],
            'fraud_detection_rate': self.metrics['fraud_detection_rate'],
            'false_positive_rate': self.metrics['false_positive_rate'],
            'total_frames_processed': self.metrics['total_frames_processed']
        }
    
    def update_metrics(self, inference_time: float, num_detections: int, 
                      tracking_accuracy: float = None, fraud_rate: float = None, 
                      fp_rate: float = None):
        """Update performance metrics"""
        self.metrics['inference_time'].append(inference_time)
        self.metrics['detections_per_frame'].append(num_detections)
        self.metrics['total_frames_processed'] += 1
        
        if tracking_accuracy is not None:
            self.metrics['tracking_accuracy'] = tracking_accuracy
        if fraud_rate is not None:
            self.metrics['fraud_detection_rate'] = fraud_rate
        if fp_rate is not None:
            self.metrics['false_positive_rate'] = fp_rate
