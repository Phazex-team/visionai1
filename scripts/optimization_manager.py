"""
Optimization Manager for Detection Pipeline
Handles ROI cropping, frame resizing, and GPU memory management
"""
import numpy as np
import cv2
import torch
from typing import Tuple, Dict, Optional
from config_models import OptimizationConfig


class OptimizationManager:
    """
    Centralized performance optimization handling.
    Applies ROI cropping, frame resizing, and GPU memory management.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.frame_count = 0
        self.roi_active = config.enable_roi_crop and config.roi_bounds is not None
    
    def prepare_frame(
        self,
        frame: np.ndarray,
        apply_roi: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Prepare frame for inference with ROI cropping and resizing.
        
        Args:
            frame: Input frame (H, W, 3)
            apply_roi: Whether to apply ROI cropping
            
        Returns:
            Tuple of (processed_frame, metadata)
            metadata includes: roi_applied, roi_bounds, crop_offset, scale_factor, original_shape
        """
        metadata = {
            'roi_applied': False,
            'roi_bounds': None,
            'crop_offset': (0, 0),
            'scale_factor': 1.0,
            'original_shape': frame.shape[:2],
            'original_dtype': frame.dtype
        }
        
        processed = frame.copy()
        
        # 1. Apply ROI cropping if enabled
        if apply_roi and self.roi_active and self.config.roi_bounds:
            processed, crop_offset = self._crop_to_roi(processed, self.config.roi_bounds)
            metadata['roi_applied'] = True
            metadata['roi_bounds'] = self.config.roi_bounds
            metadata['crop_offset'] = crop_offset
        
        # 2. Resize to max_dim if needed
        h, w = processed.shape[:2]
        if max(h, w) > self.config.max_dim:
            processed, scale = self._resize_max_dim(processed, self.config.max_dim)
            metadata['scale_factor'] *= scale
        
        return processed, metadata
    
    def _crop_to_roi(self, frame: np.ndarray, roi_bounds: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop frame to ROI region.
        
        Args:
            frame: Input frame
            roi_bounds: (x1, y1, x2, y2) in original frame coordinates
            
        Returns:
            Tuple of (cropped_frame, offset=(x1, y1))
        """
        x1, y1, x2, y2 = roi_bounds
        h, w = frame.shape[:2]
        
        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        crop = frame[y1:y2, x1:x2].copy()
        return crop, (x1, y1)
    
    def _resize_max_dim(self, frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
        """
        Resize frame so that max(height, width) == max_dim, preserving aspect ratio.
        
        Args:
            frame: Input frame
            max_dim: Maximum dimension size
            
        Returns:
            Tuple of (resized_frame, scale_factor)
        """
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame, 1.0
        
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    def map_boxes_back(
        self,
        boxes: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """
        Map detection boxes from processed frame back to original frame coordinates.
        
        Args:
            boxes: Detection boxes from model (shape: N x 4, format: [x1, y1, x2, y2])
            metadata: Metadata from prepare_frame() call
            
        Returns:
            Mapped boxes in original frame coordinates
        """
        if boxes is None or len(boxes) == 0:
            return boxes
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Undo resize (scale factor)
        scale = metadata.get('scale_factor', 1.0)
        if scale != 1.0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale
        
        # Undo ROI offset
        if metadata.get('roi_applied', False):
            offset_x, offset_y = metadata.get('crop_offset', (0, 0))
            boxes[:, [0, 2]] += offset_x
            boxes[:, [1, 3]] += offset_y
        
        return boxes
    
    def should_skip_frame(self, frame_num: int) -> bool:
        """
        Check if frame should be skipped based on skip_every_n_frames config.
        
        Args:
            frame_num: Current frame number (0-indexed)
            
        Returns:
            True if frame should be skipped
        """
        skip_interval = self.config.skip_every_n_frames
        return frame_num % skip_interval != 0
    
    def manage_gpu_memory(self, force: bool = False) -> None:
        """
        Clear GPU cache periodically to prevent out-of-memory errors.
        
        Args:
            force: Force clear regardless of frame count
        """
        self.frame_count += 1
        
        clear_interval = self.config.clear_gpu_after_n_frames
        should_clear = force or (self.frame_count % clear_interval == 0)
        
        if should_clear and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: GPU memory clear failed: {e}")
    
    def reset(self):
        """Reset internal state"""
        self.frame_count = 0


class FrameBuffer:
    """
    Circular buffer for storing frames (used for evidence recording).
    """
    
    def __init__(self, max_frames: int, fps: int = 25):
        """
        Args:
            max_frames: Maximum number of frames to keep
            fps: Frames per second (for duration calculation)
        """
        self.max_frames = max_frames
        self.fps = fps
        self.buffer = []
        self.frame_nums = []
    
    def add(self, frame: np.ndarray, frame_num: int):
        """Add frame to buffer"""
        if len(self.buffer) >= self.max_frames:
            self.buffer.pop(0)
            self.frame_nums.pop(0)
        
        self.buffer.append(frame.copy())
        self.frame_nums.append(frame_num)
    
    def get_all(self) -> Tuple[list, list]:
        """Get all frames and their frame numbers"""
        return self.buffer.copy(), self.frame_nums.copy()
    
    def get_duration_seconds(self) -> float:
        """Get duration of buffered frames in seconds"""
        if not self.frame_nums:
            return 0.0
        return (self.frame_nums[-1] - self.frame_nums[0]) / self.fps
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.frame_nums.clear()
