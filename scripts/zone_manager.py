"""
Zone Management & Hand Detection - Quick POC
"""
import numpy as np
import supervision as sv
from typing import Dict, Tuple
import cv2

class ZoneManager:
    """Manages multiple detection zones"""
    
    def __init__(self):
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.zone_annotators: Dict[str, sv.PolygonZoneAnnotator] = {}
        
    def add_zone(self, name: str, polygon: np.ndarray, color=(0, 255, 0)):
        """Add a detection zone"""
        zone = sv.PolygonZone(polygon=polygon)
        self.zones[name] = zone
        self.zone_annotators[name] = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color(color[2], color[1], color[0]),  # BGR
            thickness=2
        )
    
    def trigger_zones(self, detections) -> Dict[str, np.ndarray]:
        """Check which zones each detection is in"""
        results = {}
        for zone_name, zone in self.zones.items():
            in_zone = zone.trigger(detections=detections)
            results[zone_name] = in_zone
        return results
    
    def annotate(self, frame, zone_names=None):
        """Draw zones on frame"""
        if zone_names is None:
            zone_names = list(self.zone_annotators.keys())
        for name in zone_names:
            if name in self.zone_annotators:
                frame = self.zone_annotators[name].annotate(scene=frame)
        return frame
    
    def get_zones_for_box(self, box):
        """Return list of zones that contain this box"""
        zones_in = []
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        for zone_name, zone in self.zones.items():
            # Get zone polygon
            zone_poly = zone.polygon
            # Use point-in-polygon check
            if cv2.pointPolygonTest(zone_poly, (center_x, center_y), False) >= 0:
                zones_in.append(zone_name)
        
        return zones_in
    
    def is_in_central_counter_path(self, zones_visited):
        """Check if item followed valid checkout path (not adjacent lanes)"""
        # Valid checkout: basket → counter → scanner → pos
        # If item only in scanner/pos without counter, it's from adjacent lane
        
        if "counter" not in zones_visited:
            return False  # Never in main counter - not our transaction
        
        # Must come from basket (our checkout area), not from adjacent lane
        if "basket" not in zones_visited:
            # Could be from another lane - filter it
            if "pos" in zones_visited and "scanner" in zones_visited:
                return False  # Adjacent lane checkout
        
        return True


class SimpleHandDetector:
    """Simple hand detection using contours + edge detection"""
    
    def __init__(self, sensitivity=0.7):
        self.sensitivity = sensitivity  # 0-1, higher = more sensitive
        
    def detect_hands(self, frame) -> Tuple[list, np.ndarray]:
        """
        Simple hand detection using skin color + contour analysis
        Returns: list of hand bboxes, mask
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range (tuned for typical human skin)
        lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Also try YCrCb space (often better for skin)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 130, 75], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 140], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        mask = cv2.bitwise_or(mask, mask_ycrcb)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size - hands are medium-sized
            if 500 < area < 50000:  # Tuned bounds
                x, y, w, h = cv2.boundingRect(contour)
                hands.append((x, y, x + w, y + h))
        
        return hands, mask
    
    def check_item_occlusion(self, item_bbox, hand_bboxes) -> bool:
        """Check if hand occludes item bbox"""
        if not hand_bboxes:
            return False
        
        x1_i, y1_i, x2_i, y2_i = item_bbox
        item_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        for x1_h, y1_h, x2_h, y2_h in hand_bboxes:
            # Calculate intersection
            xi1 = max(x1_i, x1_h)
            yi1 = max(y1_i, y1_h)
            xi2 = min(x2_i, x2_h)
            yi2 = min(y2_i, y2_h)
            
            if xi2 > xi1 and yi2 > yi1:
                intersection_area = (xi2 - xi1) * (yi2 - yi1)
                overlap_ratio = intersection_area / item_area
                
                # If > 30% overlap, consider it occluded
                if overlap_ratio > 0.3:
                    return True
        
        return False
