#!/usr/bin/env python3
"""
Sequential Model Comparison with Retail Fraud Detection
Runs models one at a time for fair timing comparison and better detection
"""
import sys
import os
import gc
import json
import time
import argparse
import numpy as np
import cv2
from datetime import datetime, timedelta

# GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, os.path.dirname(__file__))

from retail_fraud_detector import RetailFraudDetector, RETAIL_TEXT_PROMPT, RETAIL_DETECTION_CLASSES
from face_blur import blur_faces

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SequentialModelComparator:
    """
    Run models sequentially for fair comparison
    Each model processes the video independently with its own fraud detector
    """
    
    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = video_path
        self.output_dir = output_dir or OUTPUT_DIR
        self.results = {}
        
        # Video info
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"[VIDEO] {video_path}")
        print(f"[VIDEO] {self.frame_width}x{self.frame_height} @ {self.fps}fps, {self.frame_count} frames")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory between models"""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
    
    def run_yoloworld(self, num_frames: int = None, skip_frames: int = 1) -> dict:
        """Run YOLOWorld model"""
        print("\n" + "="*70)
        print("Running YOLOWorld (CNN-based)")
        print("="*70)
        
        self._clear_gpu_memory()
        
        from model_yolo_world import YOLOWorldModel
        
        model = YOLOWorldModel(
            model_name="models/weights/yolov8l-worldv2.pt",
            device="cuda"
        )
        model.set_classes(RETAIL_DETECTION_CLASSES)
        model.set_thresholds(confidence=0.15, iou=0.5)
        
        result = self._process_video(model, "YOLOWorld", num_frames, skip_frames)
        
        del model
        self._clear_gpu_memory()
        
        return result
    
    def run_owlv2(self, num_frames: int = None, skip_frames: int = 1) -> dict:
        """Run OWLv2 model"""
        print("\n" + "="*70)
        print("Running OWLv2 (Vision Transformer)")
        print("="*70)
        
        self._clear_gpu_memory()
        
        try:
            from model_owlv2 import OWLv2Model
            
            model = OWLv2Model(device="cuda")
            model.set_classes(RETAIL_DETECTION_CLASSES)
            model.set_threshold(threshold=0.1)
            
            result = self._process_video(model, "OWLv2", num_frames, skip_frames)
            
            del model
            self._clear_gpu_memory()
            
            return result
        except Exception as e:
            print(f"[ERROR] OWLv2 failed: {e}")
            return None
    
    def run_yoloe(self, num_frames: int = None, skip_frames: int = 1) -> dict:
        """Run YOLOE model (Open-vocabulary)"""
        print("\n" + "="*70)
        print("Running YOLOE (Open-Vocabulary YOLO)")
        print("="*70)
        
        self._clear_gpu_memory()
        
        try:
            from model_yoloe import YOLOEModel
            
            model = YOLOEModel(
                model_name="models/weights/yoloe-11m-seg.pt",
                device="cuda"
            )
            
            if not model.model_ready:
                print("[YOLOE] Failed to load model")
                return None
            
            # IMPORTANT: set_classes MUST be called before predict for YOLOE
            model.set_classes(RETAIL_DETECTION_CLASSES)
            model.set_thresholds(confidence=0.15, iou=0.5)
            
            result = self._process_video(model, "YOLOE", num_frames, skip_frames)
            
            del model
            self._clear_gpu_memory()
            
            return result
        except Exception as e:
            print(f"[ERROR] YOLOE failed: {e}")
            return None
    
    def _process_video(self, model, model_name: str, num_frames: int = None, 
                       skip_frames: int = 1) -> dict:
        """
        Process video with a single model and fraud detection
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Initialize fraud detector for this model
        fraud_detector = RetailFraudDetector(
            fps=self.fps,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        # Hand detection for customer interaction
        from zone_manager import SimpleHandDetector
        hand_detector = SimpleHandDetector()
        
        # Tracking
        import supervision as sv
        tracker = sv.ByteTrack()
        
        # Timing
        inference_times = []
        total_detections = 0
        frames_processed = 0
        
        start_time = time.time()
        video_start_dt = datetime.now()
        
        frame_idx = 0
        while True:
            # Skip frames for faster processing
            for _ in range(skip_frames - 1):
                cap.read()
                frame_idx += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            frames_processed += 1
            
            if num_frames and frames_processed >= num_frames:
                break
            
            current_time = video_start_dt + timedelta(seconds=frame_idx / self.fps)
            
            # Run inference
            t0 = time.time()
            result = model.predict(frame)
            inference_time = (time.time() - t0) * 1000  # ms
            inference_times.append(inference_time)
            
            boxes = result.get('boxes', [])
            scores = result.get('scores', [])
            labels = result.get('labels', [])
            
            if len(boxes) > 0:
                total_detections += len(boxes)
                
                # Create supervision detections for tracking
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    confidence=np.array(scores),
                    class_id=np.zeros(len(boxes), dtype=int),
                    data={'phrases': np.array([str(l) for l in labels])}
                )
                
                # Store labels separately before NMS/tracking modifies them
                labels_array = np.array([str(l) for l in labels])
                
                # Apply NMS
                detections = detections.with_nms(threshold=0.5)
                
                # Track
                detections = tracker.update_with_detections(detections)
                
                if detections.tracker_id is not None:
                    # Detect hands
                    hand_bboxes, _ = hand_detector.detect_hands(frame)
                    
                    # Update fraud detector
                    detection_dict = {
                        'boxes': detections.xyxy.tolist(),
                        'scores': detections.confidence.tolist(),
                        'labels': labels_array.tolist() if len(labels_array) > 0 else [],
                        'tracker_ids': detections.tracker_id.tolist()
                    }
                    
                    fraud_detector.update(
                        detection_dict,
                        hand_bboxes=hand_bboxes,
                        frame_idx=frame_idx,
                        current_time=current_time
                    )
            
            if frames_processed % 100 == 0:
                avg_time = np.mean(inference_times[-100:])
                print(f"  Frame {frames_processed}: {avg_time:.1f}ms avg inference")
        
        cap.release()
        total_time = time.time() - start_time
        
        # Finalize fraud detection
        fraud_detector.finalize(current_time)
        fraud_summary = fraud_detector.get_summary()
        
        # Compile results
        result = {
            'model_name': model_name,
            'model_type': model.model_type,
            'frames_processed': frames_processed,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(frames_processed, 1),
            'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0,
            'min_inference_time_ms': np.min(inference_times) if inference_times else 0,
            'max_inference_time_ms': np.max(inference_times) if inference_times else 0,
            'total_processing_time_s': total_time,
            'fps_achieved': frames_processed / total_time if total_time > 0 else 0,
            'fraud_detection': fraud_summary
        }
        
        self.results[model_name] = result
        
        # Print summary
        print(f"\n[{model_name}] Results:")
        print(f"  Avg Inference: {result['avg_inference_time_ms']:.2f}ms")
        print(f"  Avg Detections/Frame: {result['avg_detections_per_frame']:.2f}")
        print(f"  FPS Achieved: {result['fps_achieved']:.1f}")
        
        fraud_detector.print_summary()
        
        return result
    
    def run_all_sequential(self, num_frames: int = None, skip_frames: int = 1):
        """Run all models sequentially"""
        print("\n" + "="*70)
        print("SEQUENTIAL MODEL COMPARISON")
        print("="*70)
        print(f"Video: {self.video_path}")
        print(f"Frames to process: {num_frames or 'all'}")
        print(f"Skip frames: {skip_frames}")
        print("="*70)
        
        # Run each model
        self.run_yoloworld(num_frames, skip_frames)
        self.run_owlv2(num_frames, skip_frames)
        self.run_yoloe(num_frames, skip_frames)
        
        # Print comparison
        self.print_comparison()
        
        # Export results
        self.export_results()
    
    def print_comparison(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        
        print(f"{'Model':<15} {'Type':<10} {'Avg Time':<12} {'Det/Frame':<12} {'FPS':<10} {'Fraud Events':<12}")
        print("-"*70)
        
        for name, result in self.results.items():
            if result:
                fraud_count = result['fraud_detection']['total_fraud_events']
                print(f"{name:<15} {result['model_type']:<10} "
                      f"{result['avg_inference_time_ms']:>8.2f}ms "
                      f"{result['avg_detections_per_frame']:>10.2f} "
                      f"{result['fps_achieved']:>8.1f} "
                      f"{fraud_count:>10}")
        
        print("-"*70)
        
        # Speed ranking
        print("\n⚡ SPEED RANKING (fastest to slowest):")
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if v],
            key=lambda x: x[1]['avg_inference_time_ms']
        )
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"  {i}. {name}: {result['avg_inference_time_ms']:.2f}ms")
        
        # Detection ranking
        print("\n🎯 DETECTION RANKING (most detections):")
        sorted_by_det = sorted(
            [(k, v) for k, v in self.results.items() if v],
            key=lambda x: x[1]['avg_detections_per_frame'],
            reverse=True
        )
        for i, (name, result) in enumerate(sorted_by_det, 1):
            print(f"  {i}. {name}: {result['avg_detections_per_frame']:.2f} det/frame")
    
    def export_results(self):
        """Export results to JSON"""
        output_file = os.path.join(self.output_dir, "sequential_comparison_results.json")
        
        # Convert for JSON serialization
        export_data = {}
        for name, result in self.results.items():
            if result:
                export_data[name] = {
                    k: v for k, v in result.items()
                    if not isinstance(v, (np.ndarray, np.floating, np.integer))
                }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n[EXPORT] Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Sequential Model Comparison with Fraud Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to process")
    parser.add_argument("--skip", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["all", "yoloworld", "owlv2", "yoloe"],
                        help="Which model to run")
    
    args = parser.parse_args()
    
    # Make video path absolute
    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(BASE_DIR, video_path)
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    comparator = SequentialModelComparator(video_path)
    
    if args.model == "all":
        comparator.run_all_sequential(args.frames, args.skip)
    elif args.model == "yoloworld":
        comparator.run_yoloworld(args.frames, args.skip)
    elif args.model == "owlv2":
        comparator.run_owlv2(args.frames, args.skip)
    elif args.model == "yoloe":
        comparator.run_yoloe(args.frames, args.skip)


if __name__ == "__main__":
    main()
