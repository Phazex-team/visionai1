"""
Factory Pattern for Model Instantiation and Configuration
Centralizes model creation with unified configuration
"""
import os
import sys
from typing import Optional
from config_models import DetectionConfig, OptimizationConfig, ModelType

# Add base directory to sys.path for package imports
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)


class ModelFactory:
    """
    Factory for creating detection models with unified configuration.
    Handles all model instantiation, path resolution, and config application.
    """
    
    # Model imports (lazy loaded to avoid circular imports)
    _model_classes = {
        'yoloworld': None,
        'owlv2': None,
        'yoloe': None,
        'groundingdino': None
    }
    
    @classmethod
    def _ensure_imports(cls):
        """Lazy load model classes from models/ subdirectories"""
        if cls._model_classes['yoloworld'] is None:
            try:
                from models.v2_extended.model_yolo_world import YOLOWorldModel
                cls._model_classes['yoloworld'] = YOLOWorldModel
            except ImportError as e:
                print(f"Warning: Could not import YOLOWorldModel: {e}")
        
        if cls._model_classes['owlv2'] is None:
            try:
                from models.v2_extended.model_owlv2 import OWLv2Model
                cls._model_classes['owlv2'] = OWLv2Model
            except ImportError as e:
                print(f"Warning: Could not import OWLv2Model: {e}")
        
        if cls._model_classes['yoloe'] is None:
            try:
                from models.v3_optimized.model_yoloe import YOLOEModel
                cls._model_classes['yoloe'] = YOLOEModel
            except ImportError as e:
                print(f"Warning: Could not import YOLOEModel: {e}")
        
        if cls._model_classes['groundingdino'] is None:
            try:
                from models.v1_base.model_grounding_dino import GroundingDINOModel
                cls._model_classes['groundingdino'] = GroundingDINOModel
            except ImportError as e:
                print(f"Warning: Could not import GroundingDINOModel: {e}")
    
    @staticmethod
    def _resolve_model_path(model_name: str, provided_path: Optional[str] = None) -> Optional[str]:
        """
        Resolve model path from provided path or default locations
        
        Args:
            model_name: Name of the model
            provided_path: User-provided path (takes precedence)
            
        Returns:
            Resolved path or None if not found
        """
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # Default paths
        default_paths = {
            'yoloworld': [
                'models/weights/yolov8l-worldv2.pt',
                'weights/yolov8l-worldv2.pt',
                'scripts/yolov8l-worldv2.pt'
            ],
            'yoloe': [
                'models/weights/yoloe-11m-seg.pt',
                'weights/yoloe-11m-seg.pt',
                'scripts/yoloe-11m-seg.pt'
            ],
            'groundingdino': [
                'models/GroundingDINO_SwinT_OGC.py',
                'weights/groundingdino_swint_ogc.pth'
            ],
            'owlv2': [
                None  # OWLv2 downloads model automatically
            ]
        }
        
        for path in default_paths.get(model_name, []):
            if path and os.path.exists(path):
                return path
        
        return None
    
    @classmethod
    def create_model(
        cls,
        detection_config: DetectionConfig,
        optimization_config: OptimizationConfig
    ):
        """
        Create and configure a detection model with unified settings.
        
        Args:
            detection_config: DetectionConfig with model name, thresholds, device
            optimization_config: OptimizationConfig with performance tuning
            
        Returns:
            Configured model instance
            
        Raises:
            ValueError: If model cannot be created or configured
        """
        cls._ensure_imports()
        
        model_name = detection_config.model_name.lower()
        
        print(f"\n{'='*60}")
        print(f"Creating Model: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Resolve model path
        model_path = cls._resolve_model_path(model_name, detection_config.model_path)
        if detection_config.model_path and model_path:
            print(f"  Model path: {model_path}")
        
        # Create model instance based on type
        if model_name == 'yoloworld':
            model = cls._create_yoloworld(detection_config, model_path)
        
        elif model_name == 'owlv2':
            model = cls._create_owlv2(detection_config)
        
        elif model_name == 'yoloe':
            model = cls._create_yoloe(detection_config, model_path)
        
        elif model_name == 'groundingdino':
            model = cls._create_groundingdino(detection_config, model_path, detection_config.weights_path)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply unified detection config
        print(f"  Applying detection config...")
        model.set_detection_config(detection_config)
        
        # Store optimization config in model metadata
        model.optimization_config = optimization_config
        model.metadata.update({
            'roi': optimization_config.roi_bounds,
            'skip_every_n': optimization_config.skip_every_n_frames,
            'max_dim': optimization_config.max_dim,
            'enable_roi_crop': optimization_config.enable_roi_crop
        })
        
        print(f"  ✅ {model_name.upper()} ready")
        print(f"  Confidence threshold: {detection_config.confidence_threshold}")
        print(f"  IOU threshold: {detection_config.iou_threshold}")
        if optimization_config.enable_roi_crop:
            print(f"  ROI cropping enabled: {optimization_config.roi_bounds}")
        print(f"  Device: {detection_config.device}")
        print(f"{'='*60}\n")
        
        return model
    
    @classmethod
    def _create_yoloworld(cls, config: DetectionConfig, model_path: Optional[str]):
        """Create YOLOWorld model"""
        ModelClass = cls._model_classes['yoloworld']
        if ModelClass is None:
            raise ImportError("YOLOWorldModel not available")
        
        if model_path:
            return ModelClass(model_name=model_path, device=config.device)
        else:
            return ModelClass(device=config.device)
    
    @classmethod
    def _create_owlv2(cls, config: DetectionConfig):
        """Create OWLv2 model"""
        ModelClass = cls._model_classes['owlv2']
        if ModelClass is None:
            raise ImportError("OWLv2Model not available")
        
        return ModelClass(device=config.device)
    
    @classmethod
    def _create_yoloe(cls, config: DetectionConfig, model_path: Optional[str]):
        """Create YOLOE model"""
        ModelClass = cls._model_classes['yoloe']
        if ModelClass is None:
            raise ImportError("YOLOEModel not available")
        
        if model_path:
            return ModelClass(model_name=model_path, device=config.device)
        else:
            return ModelClass(device=config.device)
    
    @classmethod
    def _create_groundingdino(cls, config: DetectionConfig, model_path: Optional[str], weights_path: Optional[str]):
        """Create GroundingDINO model"""
        ModelClass = cls._model_classes['groundingdino']
        if ModelClass is None:
            raise ImportError("GroundingDINOModel not available")
        
        return ModelClass(
            model_path=model_path,
            weights_path=weights_path,
            device=config.device
        )


class ModelRegistry:
    """
    Registry for keeping track of created models and their configs.
    Useful for model switching and comparison.
    """
    
    def __init__(self):
        self.models = {}
        self.configs = {}
    
    def register_model(self, name: str, model, config: DetectionConfig, opt_config: OptimizationConfig):
        """Register a model with its configuration"""
        self.models[name] = model
        self.configs[name] = {
            'detection': config,
            'optimization': opt_config
        }
        print(f"Registered model: {name}")
    
    def get_model(self, name: str):
        """Get a registered model by name"""
        return self.models.get(name)
    
    def get_config(self, name: str):
        """Get a registered model's configuration"""
        return self.configs.get(name)
    
    def list_models(self):
        """List all registered models"""
        return list(self.models.keys())
    
    def remove_model(self, name: str):
        """Remove a model from registry"""
        if name in self.models:
            del self.models[name]
            del self.configs[name]
            print(f"Removed model: {name}")
