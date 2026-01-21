"""
Retail Fraud Detection System v2
Proper flow: Grace period → Touch detection → Scan check → Exit confirmation → Fraud

FLOW:
1. Startup grace period (ignore items already on counter)
2. Track item states: NEW → TOUCHED → ON_COUNTER → SCANNED/LEFT_COUNTER
3. Only flag fraud when: item left counter + not scanned + moving toward exit
4. Evidence recording: save before/after fraud confirmation
"""
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class ItemState(Enum):
    """Item lifecycle states"""
    NEW = "new"                    # Just detected
    STATIONARY = "stationary"     # On counter at startup (ignore)
    TOUCHED = "touched"           # Item was picked up/moved
    ON_COUNTER = "on_counter"     # Item on counter after being touched
    IN_TROLLEY = "in_trolley"     # Item in trolley/cart
    IN_SCANNER = "in_scanner"     # Item in scanner zone
    SCANNED = "scanned"           # Item properly scanned
    LEFT_COUNTER = "left_counter" # Item left counter (check scan status)
    CONCEALED = "concealed"       # Item disappeared near person (pocket/clothing)
    EXITING = "exiting"           # Item moving toward exit
    FRAUD_CONFIRMED = "fraud"     # Unscanned item confirmed as fraud


class FraudType(Enum):
    MISSED_SCAN = "missed_scan"
    QUICK_PASS = "quick_pass"
    CASHIER_SKIP = "cashier_skip"
    HAND_CONCEALMENT = "hand_concealment"
    POCKET_CONCEALMENT = "pocket_concealment"  # Item hidden in pocket/clothing
    TROLLEY_HIDDEN = "trolley_hidden"
    PACKING_AREA_UNSCANNED = "packing_area_unscanned"  # Item in packing area without scan


class WarningType(Enum):
    """Warning types (not confirmed fraud yet)"""
    PACKING_AREA_WARNING = "packing_area_warning"  # Item in packing area too long without scan


@dataclass
class TrackedItemV2:
    """Enhanced item tracking with state machine"""
    track_id: int
    label: str
    confidence: float
    
    # State
    state: ItemState = ItemState.NEW
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    # Position tracking
    positions: List[Tuple[float, float]] = field(default_factory=list)
    last_bbox: Tuple[int, int, int, int] = None
    
    # Movement tracking
    total_movement: float = 0
    movement_threshold: float = 50  # pixels - movement to be considered "touched"
    
    # Zone timing
    counter_entry_frame: int = 0
    counter_exit_frame: int = 0
    scanner_frames: int = 0  # frames in scanner zone
    trolley_frames: int = 0  # frames in trolley zone
    packing_area_frames: int = 0  # frames in packing area
    packing_area_entry_frame: int = 0  # when item entered packing area
    packing_area_warning_sent: bool = False  # warning already sent
    
    # Touch/Movement tracking
    touched_frame: int = 0  # frame when item was first touched/moved significantly
    
    # Visibility tracking (for concealment detection)
    frames_not_visible: int = 0
    last_visible_frame: int = 0
    near_person_when_disappeared: bool = False
    concealment_frame: int = 0
    
    # Scan status
    is_scanned: bool = False
    scan_confirmed_frame: int = 0
    
    # Fraud
    fraud_type: Optional[FraudType] = None
    fraud_frame: int = 0
    
    # Ownership / ROI context
    owner_id: Optional[int] = None
    owner_role: str = "unknown"
    ownership_history: List[Dict[str, Any]] = field(default_factory=list)
    last_known_zone: Optional[str] = None
    last_seen_in_roi: bool = False
    entered_checkout: bool = False
    last_owner_update: int = 0
    
    def add_position(self, bbox, frame_num):
        """Track position and calculate movement"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Calculate movement from last position
        if self.positions:
            last_x, last_y = self.positions[-1]
            movement = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
            self.total_movement += movement
        
        self.positions.append((cx, cy))
        self.last_bbox = bbox
        self.last_seen_frame = frame_num
    
    def was_touched(self) -> bool:
        """Check if item has moved enough to be considered touched"""
        return self.total_movement > self.movement_threshold
    
    def get_movement_direction(self) -> Tuple[float, float]:
        """Get average movement direction (for exit detection)"""
        if len(self.positions) < 2:
            return (0, 0)
        
        # Use the most recent positions (up to 5) for stability
        recent = self.positions[-min(5, len(self.positions)):]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx, dy)


@dataclass
class PersonTrack:
    """Lightweight person track for ownership association"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    role: str = "unknown"  # cashier | customer | unknown
    last_seen_frame: int = 0
    zones: List[str] = field(default_factory=list)


class FraudDetectorV2:
    """
    Fraud detector with proper detection flow
    """
    
    def __init__(self, fps=25, frame_width=1280, frame_height=720):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Items tracking
        self.items: Dict[int, TrackedItemV2] = {}
        
        # Timing
        self.start_frame = 0
        self.current_frame = 0
        self.startup_grace_frames = int(fps * 5)  # 5 seconds grace period
        self.is_in_grace_period = True
        
        # Scanner settings
        self.min_scanner_frames = int(fps * 0.5)  # 0.5 seconds in scanner = scanned
        
        # Packing area settings
        self.packing_area_warning_seconds = 5  # Warn if item in packing area > 5 seconds without scan
        self.packing_area_warning_frames = int(fps * self.packing_area_warning_seconds)
        
        # Events
        self.fraud_events: List[Dict] = []
        self.warning_events: List[Dict] = []  # Warnings (not confirmed fraud)
        self.logged_fraud_ids: Set[int] = set()
        self.logged_warning_ids: Set[int] = set()
        
        # Counters (for UI)
        self.counter_items: Set[int] = set()
        self.scanned_item_ids: Set[int] = set()
        self.missed_scans: int = 0
        self.quick_pass_events: int = 0
        self.cashier_skip_events: int = 0
        self.hand_pickups: int = 0
        self.trolley_hidden: int = 0
        self.pocket_concealment: int = 0
        self.packing_area_warnings: int = 0
        self.packing_area_frauds: int = 0
        
        # Person tracking (for concealment detection)
        self.person_positions: List[Tuple[int, int, int, int]] = []  # Current frame person bboxes
        self.person_tracks: Dict[int, PersonTrack] = {}  # Current frame person tracks with roles
        self.person_labels = {'person', 'customer', 'human', 'cashier'}
        self.ignore_item_labels = {'person', 'customer', 'human', 'hand', 'scanner'}
        self.concealment_distance_threshold: int = 100  # pixels - item must be this close to person
        self.concealment_frames_threshold: int = int(fps * 1.5)  # 1.5 seconds missing = concealed
        self.owner_distance_threshold: int = 180  # pixels - max distance to associate item to person
        
        # ROI / zone enforcement
        self.checkout_zone_priority = [
            'counter', 'scanner', 'packing_area', 'customer_area',
            'trolley', 'basket', 'cashier', 'pos'
        ]
        self.exit_zone_names = ['exit']
        self.zone_priority = self.checkout_zone_priority + self.exit_zone_names
        
        # Evidence recording
        self.frame_buffer: deque = deque(maxlen=int(fps * 3))  # 3 seconds buffer
        self.pending_evidence: List[Dict] = []  # Frauds waiting for after-frames
        
        # Zones
        self.zones = {}
        self._setup_default_zones()
        self._load_zones_config()
        
        print(f"[FRAUD DETECTOR v2] Initialized")
        print(f"[FRAUD DETECTOR v2] Startup grace period: {self.startup_grace_frames / fps:.1f} seconds")
        print(f"[FRAUD DETECTOR v2] Scanner min time: {self.min_scanner_frames / fps:.1f} seconds")
    
    def _setup_default_zones(self):
        """Default zone layout - covers all important areas"""
        self.zones = {
            'counter': np.array([[300, 200], [900, 200], [900, 400], [300, 400]]),
            'scanner': np.array([[500, 250], [750, 250], [750, 350], [500, 350]]),
            'pos': np.array([[450, 200], [600, 200], [600, 350], [450, 350]]),
            'exit': np.array([[900, 400], [1280, 400], [1280, 720], [900, 720]]),
            'customer_area': np.array([[100, 400], [600, 400], [600, 700], [100, 700]]),
            'trolley': np.array([[150, 450], [500, 450], [500, 650], [150, 650]]),
            'basket': np.array([[200, 500], [450, 500], [450, 600], [200, 600]]),
            'cashier': np.array([[300, 100], [600, 100], [600, 250], [300, 250]]),
            'packing_area': np.array([[700, 300], [950, 300], [950, 500], [700, 500]]),
        }
    
    def _load_zones_config(self):
        """Load zones from config file"""
        import os
        import json
        
        config_paths = [
            'zones_config.json',
            '../zones_config.json',
            os.path.join(os.path.dirname(__file__), '..', 'zones_config.json')
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        config = json.load(f)
                    
                    for zone_name, zone_data in config.get('zones', {}).items():
                        # Handle both formats:
                        # 1. Direct polygon: {"counter": [[x,y], [x,y], ...]}
                        # 2. Nested format: {"counter": {"polygon": [[x,y], ...]}}
                        if isinstance(zone_data, list):
                            self.zones[zone_name] = np.array(zone_data, np.int32)
                        elif isinstance(zone_data, dict) and 'polygon' in zone_data:
                            self.zones[zone_name] = np.array(zone_data['polygon'], np.int32)
                    
                    print(f"[FRAUD DETECTOR v2] Loaded zones: {list(self.zones.keys())}")
                    return
                except Exception as e:
                    print(f"[FRAUD DETECTOR v2] Error loading zones: {e}")
    
    def configure_zones(self, zones_dict: Dict[str, np.ndarray]):
        """Configure zones from external source"""
        self.zones.update(zones_dict)
    
    def _point_in_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        """Check if point is inside a zone"""
        if zone_name not in self.zones:
            return False
        return cv2.pointPolygonTest(self.zones[zone_name], point, False) >= 0
    
    def _get_zones_for_point(self, point: Tuple[float, float]) -> List[str]:
        """Get all zones containing a point"""
        zones = []
        for name in self.zones:
            if self._point_in_zone(point, name):
                zones.append(name)
        return zones
    
    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Return center point of a bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _is_point_in_checkout(self, point: Tuple[float, float]) -> Tuple[bool, Optional[str]]:
        """
        Check if a point falls inside any configured checkout ROI zone.
        Returns (True, zone_name) for the first matching zone in priority order.
        """
        for zone_name in self.checkout_zone_priority:
            if zone_name in self.zones and self._point_in_zone(point, zone_name):
                return True, zone_name
        return False, None
    
    def _select_primary_zone(self, zones: List[str]) -> Optional[str]:
        """Pick the most relevant zone for a set of memberships"""
        for name in self.zone_priority:
            if name in zones:
                return name
        return zones[0] if zones else None
    
    def _infer_person_role(self, zones: List[str]) -> str:
        """Infer whether a person is cashier or customer based on zones"""
        if 'cashier' in zones or 'pos' in zones:
            return 'cashier'
        if any(z in zones for z in ['customer_area', 'trolley', 'basket']):
            return 'customer'
        return 'unknown'
    
    def _is_moving_toward_exit(self, item: TrackedItemV2) -> bool:
        """Check if item is moving toward exit area"""
        dx, dy = item.get_movement_direction()
        
        # Exit is typically right side of frame
        # Positive dx = moving right = toward exit
        return dx > 20  # Moving right significantly
    
    def _is_near_person(self, point: Tuple[float, float]) -> bool:
        """Check if a point is near any detected person"""
        px, py = point
        for person_bbox in self.person_positions:
            x1, y1, x2, y2 = person_bbox
            # Check if point is within or near person bbox
            # Expand bbox by threshold
            if (x1 - self.concealment_distance_threshold <= px <= x2 + self.concealment_distance_threshold and
                y1 - self.concealment_distance_threshold <= py <= y2 + self.concealment_distance_threshold):
                return True
        return False
    
    def _assign_owner(self, item: TrackedItemV2, point: Tuple[float, float]):
        """Associate an item with the closest person track in the checkout ROI"""
        # Keep recent assignment to avoid flicker
        if item.owner_id is not None and (self.current_frame - item.last_owner_update) <= 5:
            return
        
        best_person = None
        best_dist = float('inf')
        
        for person in self.person_tracks.values():
            px, py = self._bbox_center(person.bbox)
            dist = np.sqrt((px - point[0])**2 + (py - point[1])**2)
            if dist < best_dist and dist <= self.owner_distance_threshold:
                best_dist = dist
                best_person = person
        
        if best_person:
            item.owner_id = best_person.track_id
            item.owner_role = best_person.role
            item.last_owner_update = self.current_frame
            item.ownership_history.append({
                'frame': self.current_frame,
                'owner_id': best_person.track_id,
                'owner_role': best_person.role,
                'distance': round(best_dist, 2)
            })
    
    def _update_person_tracks(self, detections, frame_idx: int):
        """Update current person positions/tracks from detections"""
        self.person_positions = []
        self.person_tracks = {}
        
        if detections is None:
            return
        
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        tracker_ids = detections.get('tracker_ids', list(range(len(boxes))))
        
        for i, box in enumerate(boxes):
            label = str(labels[i]).lower() if i < len(labels) else ""
            if label not in self.person_labels:
                continue
            
            cx, cy = self._bbox_center(box)
            in_checkout, zone_name = self._is_point_in_checkout((cx, cy))
            zones = self._get_zones_for_point((cx, cy))
            
            # Ignore persons fully outside checkout + exit ROI to avoid cross-lane noise
            if not in_checkout and 'exit' not in zones:
                continue
            
            track_id = tracker_ids[i] if i < len(tracker_ids) else i
            person = PersonTrack(
                track_id=track_id,
                bbox=tuple(box),
                role=self._infer_person_role(zones),
                last_seen_frame=frame_idx,
                zones=zones
            )
            self.person_tracks[track_id] = person
            self.person_positions.append(tuple(box))
    
    def process_frame(self, frame, frame_num: int, detections: Dict, pos_match: Dict = None) -> Dict:
        """
        Process a frame for fraud detection.
        Wrapper around update() method for pipeline compatibility.
        
        Args:
            frame: Current video frame (numpy array)
            frame_num: Frame number
            detections: Detection results with boxes, labels, scores
            pos_match: POS matching results (optional)
        
        Returns:
            Dictionary with fraud_events and other status info
        """
        # Call the update method
        fraud_events = self.update(detections, frame_num, frame)
        
        return {
            'fraud_events': fraud_events,
            'frame_num': frame_num,
            'counter_items': len(self.counter_items),
            'scanned_items': len(self.scanned_item_ids),
            'missed_scans': self.missed_scans,
            'hand_pickups': self.hand_pickups,
            'trolley_hidden': self.trolley_hidden,
            'pocket_concealment': self.pocket_concealment,
            'pos_match': pos_match,
            'is_grace_period': self.is_in_grace_period
        }

    
    def update(self, detections, frame_idx: int = 0, frame: np.ndarray = None) -> List[Dict]:
        """
        Update fraud detection with new frame
        
        Args:
            detections: Detection results
            frame_idx: Current frame number
            frame: Current frame (for evidence recording)
        
        Returns:
            List of new fraud events
        """
        self.current_frame = frame_idx
        
        # Store frame in buffer for evidence
        if frame is not None:
            self.frame_buffer.append({
                'frame': frame.copy(),
                'frame_num': frame_idx
            })
        
        # Update person positions for concealment detection
        self._update_person_tracks(detections, frame_idx)
        
        # Check if still in grace period
        if frame_idx < self.startup_grace_frames:
            if self.is_in_grace_period:
                # Mark all currently visible items as stationary (ignore them)
                self._mark_startup_items(detections)
            return []
        
        if self.is_in_grace_period:
            self.is_in_grace_period = False
            print(f"[FRAUD DETECTOR v2] Grace period ended at frame {frame_idx}")
        
        # Get currently visible item IDs
        visible_item_ids = self._get_visible_item_ids(detections)
        
        # Check for concealment (items that disappeared near person)
        concealment_events = self._check_concealment(visible_item_ids, frame_idx)
        
        # Process detections
        new_events = self._process_detections(detections, frame_idx)
        new_events.extend(concealment_events)
        
        # Check for fraud confirmations
        fraud_events = self._check_fraud_confirmations(frame_idx)
        new_events.extend(fraud_events)
        
        return new_events
    
    def _get_visible_item_ids(self, detections) -> Set[int]:
        """Get set of currently visible item track IDs"""
        visible = set()
        
        if detections is None:
            return visible
        
        boxes = detections.get('boxes', [])
        tracker_ids = detections.get('tracker_ids', list(range(len(boxes))))
        labels = detections.get('labels', [])
        
        for i, box in enumerate(boxes):
            track_id = tracker_ids[i] if i < len(tracker_ids) else i
            label = str(labels[i]).lower() if i < len(labels) else ""
            
            # Only count non-person items
            if label in self.ignore_item_labels:
                continue
            
            item = self.items.get(track_id)
            if item and item.entered_checkout:
                visible.add(track_id)
                continue
            
            # If new item, only consider it visible when inside checkout ROI
            cx, cy = self._bbox_center(box)
            in_checkout, _ = self._is_point_in_checkout((cx, cy))
            if in_checkout:
                visible.add(track_id)
        
        return visible
    
    def _check_concealment(self, visible_item_ids: Set[int], frame_idx: int) -> List[Dict]:
        """
        Check for items that disappeared near a person (pocket/clothing concealment)
        
        STRICT RULES - Only trigger fraud when:
        1. Item disappeared near person
        2. Item has no scan record
        3. Item was last seen moving toward exit OR near exit zone
        """
        events = []
        
        for track_id, item in self.items.items():
            # Skip already processed items
            if item.state in [ItemState.STATIONARY, ItemState.SCANNED, ItemState.FRAUD_CONFIRMED, ItemState.CONCEALED]:
                continue
            
            # Ignore items that never entered the checkout ROI
            if not item.entered_checkout:
                continue
            
            # Skip if already logged
            if track_id in self.logged_fraud_ids:
                continue
            
            # Check if item is visible
            if track_id in visible_item_ids:
                # Item is visible - reset concealment tracking
                item.frames_not_visible = 0
                item.last_visible_frame = frame_idx
                
                # Check if item was near person when last seen
                if item.positions:
                    item.near_person_when_disappeared = self._is_near_person(item.positions[-1])
            else:
                # Item not visible
                item.frames_not_visible += 1
                
                # Check for concealment: 
                # 1. Item was touched/on counter
                # 2. Near person when disappeared
                # 3. Missing for threshold time
                # 4. NOT scanned
                # 5. Was moving toward exit OR near exit zone
                if (item.state in [ItemState.TOUCHED, ItemState.ON_COUNTER, ItemState.IN_TROLLEY, ItemState.LEFT_COUNTER] and
                    item.was_touched() and
                    item.near_person_when_disappeared and
                    item.frames_not_visible >= self.concealment_frames_threshold and
                    not item.is_scanned):
                    
                    # Check if item was moving toward exit or near exit when last seen
                    was_near_exit = False
                    if item.positions:
                        last_pos = item.positions[-1]
                        was_near_exit = self._point_in_zone(last_pos, 'exit') or self._is_moving_toward_exit(item)
                    
                    if was_near_exit:
                        # FRAUD: Item concealed while heading to exit
                        item.state = ItemState.CONCEALED
                        item.fraud_type = FraudType.POCKET_CONCEALMENT
                        item.fraud_frame = frame_idx
                        item.concealment_frame = frame_idx
                        self.logged_fraud_ids.add(track_id)
                        self.pocket_concealment += 1
                        
                        event = {
                            'type': 'pocket_concealment',
                            'track_id': track_id,
                            'label': item.label,
                            'confidence': item.confidence,
                            'frame': frame_idx,
                            'frame_num': frame_idx,
                            'bbox': item.last_bbox,
                            'last_seen_frame': item.last_visible_frame,
                            'frames_missing': item.frames_not_visible,
                            'roi_zone': item.last_known_zone,
                            'roi_confirmed': item.last_seen_in_roi or was_near_exit,
                            'owner_id': item.owner_id,
                            'owner_role': item.owner_role,
                            'ownership_history': item.ownership_history,
                            'message': f"FRAUD: '{item.label}' concealed near exit (missing for {item.frames_not_visible/self.fps:.1f}s)"
                        }
                        events.append(event)
                        self.fraud_events.append(event)
                        print(f"[FRAUD] ⚠️ POCKET CONCEALMENT: Item {track_id} '{item.label}' - concealed near exit!")
        
        return events
    
    def _mark_startup_items(self, detections):
        """Mark items visible during startup as stationary (ignore them)"""
        if detections is None:
            return
        
        boxes = detections.get('boxes', [])
        tracker_ids = detections.get('tracker_ids', list(range(len(boxes))))
        labels = detections.get('labels', [])
        scores = detections.get('scores', [])
        
        for i, box in enumerate(boxes):
            track_id = tracker_ids[i] if i < len(tracker_ids) else i
            label = str(labels[i]).lower() if i < len(labels) else "item"
            
            # Skip non-item labels and anything outside the checkout ROI
            if label in self.ignore_item_labels:
                continue
            in_checkout, zone_name = self._is_point_in_checkout(self._bbox_center(box))
            if not in_checkout:
                continue
            
            if track_id not in self.items:
                self.items[track_id] = TrackedItemV2(
                    track_id=track_id,
                    label=str(labels[i]) if i < len(labels) else "item",
                    confidence=float(scores[i]) if i < len(scores) else 0.5,
                    state=ItemState.STATIONARY,  # Mark as stationary - will be ignored
                    first_seen_frame=self.current_frame
                )
                self.items[track_id].add_position(box, self.current_frame)
    
    def _process_detections(self, detections, frame_idx: int) -> List[Dict]:
        """Process detections and update item states"""
        events = []
        
        if detections is None:
            return events
        
        boxes = detections.get('boxes', [])
        tracker_ids = detections.get('tracker_ids', list(range(len(boxes))))
        labels = detections.get('labels', [])
        scores = detections.get('scores', [])
        
        for i, box in enumerate(boxes):
            track_id = tracker_ids[i] if i < len(tracker_ids) else i
            label = str(labels[i]) if i < len(labels) else "item"
            label_l = label.lower()
            score = float(scores[i]) if i < len(scores) else 0.5
            
            # Ignore non-item detections (people, hands, scanner hardware)
            if label_l in self.ignore_item_labels:
                continue
            
            # Get center point
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            zones = self._get_zones_for_point((cx, cy))
            primary_zone = self._select_primary_zone(zones) or "unknown"
            in_checkout, _ = self._is_point_in_checkout((cx, cy))
            in_exit_zone = 'exit' in zones
            
            # Create new item if needed
            if track_id not in self.items:
                self.items[track_id] = TrackedItemV2(
                    track_id=track_id,
                    label=label,
                    confidence=score,
                    state=ItemState.NEW,
                    first_seen_frame=frame_idx
                )
            
            item = self.items[track_id]
            
            # Guardrail: do not run fraud logic for objects that never entered checkout ROI
            if (not item.entered_checkout) and (not in_checkout) and (not in_exit_zone):
                item.last_bbox = box
                item.last_seen_frame = frame_idx
                item.last_known_zone = primary_zone
                item.last_seen_in_roi = False
                continue
            
            # When an item first crosses into ROI, reset movement to avoid counting outside motion
            if in_checkout and not item.entered_checkout:
                item.entered_checkout = True
                item.positions = []
                item.total_movement = 0
            
            item.add_position(box, frame_idx)
            item.last_known_zone = primary_zone
            item.last_seen_in_roi = in_checkout or item.last_seen_in_roi
            item.last_seen_frame = frame_idx
            
            # Associate to nearest person inside ROI
            if in_checkout:
                self._assign_owner(item, (cx, cy))
            
            # Skip items marked as stationary (from startup)
            if item.state == ItemState.STATIONARY:
                continue
            
            # ========== STATE MACHINE ==========
            
            # NEW → TOUCHED (if moved significantly) or IN_TROLLEY (if in trolley zone)
            if item.state == ItemState.NEW:
                if 'trolley' in zones:
                    item.state = ItemState.IN_TROLLEY
                    item.trolley_frames += 1
                    print(f"[FRAUD] Item {track_id} '{label}' detected IN_TROLLEY")
                elif in_checkout and item.was_touched():
                    item.state = ItemState.TOUCHED
                    item.touched_frame = frame_idx  # Record when item was first touched
                    self._assign_owner(item, (cx, cy))
                    print(f"[FRAUD] Item {track_id} '{label}' was TOUCHED (moved {item.total_movement:.0f}px)")
            
            # IN_TROLLEY → TOUCHED (if picked up from trolley)
            elif item.state == ItemState.IN_TROLLEY:
                if 'trolley' in zones:
                    item.trolley_frames += 1
                elif in_checkout and item.was_touched():
                    item.state = ItemState.TOUCHED
                    item.touched_frame = frame_idx  # Record when item was picked up
                    self._assign_owner(item, (cx, cy))
                    print(f"[FRAUD] Item {track_id} '{label}' picked up from trolley")
            
            # TOUCHED → ON_COUNTER (if in counter zone)
            elif item.state == ItemState.TOUCHED:
                if 'counter' in zones:
                    item.state = ItemState.ON_COUNTER
                    item.counter_entry_frame = frame_idx
                    self.counter_items.add(track_id)
                    self._assign_owner(item, (cx, cy))
                    print(f"[FRAUD] Item {track_id} '{label}' placed ON_COUNTER")
            
            # ON_COUNTER → IN_SCANNER or LEFT_COUNTER
            elif item.state == ItemState.ON_COUNTER:
                if 'scanner' in zones:
                    item.state = ItemState.IN_SCANNER
                    item.scanner_frames += 1
                elif 'counter' not in zones:
                    # Left counter without going through scanner
                    item.state = ItemState.LEFT_COUNTER
                    item.counter_exit_frame = frame_idx
                    print(f"[FRAUD] Item {track_id} '{label}' LEFT_COUNTER (scanner time: {item.scanner_frames/self.fps:.2f}s)")
                    
                    # Cashier skip: cashier-owned item leaves counter without scan
                    if (not item.is_scanned and item.owner_role == 'cashier' and 
                        item.scanner_frames < self.min_scanner_frames and
                        track_id not in self.logged_fraud_ids):
                        item.state = ItemState.FRAUD_CONFIRMED
                        item.fraud_type = FraudType.CASHIER_SKIP
                        item.fraud_frame = frame_idx
                        self.logged_fraud_ids.add(track_id)
                        self.cashier_skip_events += 1
                        
                        event = {
                            'type': item.fraud_type.value,
                            'track_id': track_id,
                            'label': label,
                            'confidence': score,
                            'frame': frame_idx,
                            'frame_num': frame_idx,
                            'start_frame': item.touched_frame if item.touched_frame > 0 else frame_idx,
                            'bbox': item.last_bbox,
                            'scanner_time': item.scanner_frames / self.fps,
                            'roi_zone': item.last_known_zone,
                            'roi_confirmed': item.entered_checkout,
                            'owner_id': item.owner_id,
                            'owner_role': item.owner_role,
                            'ownership_history': item.ownership_history,
                            'message': f"FRAUD CONFIRMED: CASHIER skipped scan for '{label}' leaving counter"
                        }
                        events.append(event)
                        self.fraud_events.append(event)
                        print(f"[FRAUD] ❌ FRAUD (cashier_skip): Item {track_id} '{label}' left counter with no scan")
            
            # IN_SCANNER → SCANNED or back to ON_COUNTER
            elif item.state == ItemState.IN_SCANNER:
                if 'scanner' in zones:
                    item.scanner_frames += 1
                    
                    # Check if scanned
                    if item.scanner_frames >= self.min_scanner_frames and not item.is_scanned:
                        item.is_scanned = True
                        item.state = ItemState.SCANNED
                        item.scan_confirmed_frame = frame_idx
                        self.scanned_item_ids.add(track_id)
                        self._assign_owner(item, (cx, cy))
                        print(f"[FRAUD] Item {track_id} '{label}' SCANNED ✓")
                else:
                    # Left scanner - go back to counter or left counter
                    if 'counter' in zones:
                        item.state = ItemState.ON_COUNTER
                    else:
                        item.state = ItemState.LEFT_COUNTER
                        item.counter_exit_frame = frame_idx
            
            # SCANNED → just track position
            elif item.state == ItemState.SCANNED:
                pass  # Item is OK, just track it

            # LEFT_COUNTER → check if moving to exit (FRAUD)
            elif item.state == ItemState.LEFT_COUNTER:
                if not item.is_scanned and track_id not in self.logged_fraud_ids:
                    # Check if moving toward exit
                    if self._is_moving_toward_exit(item):
                        item.state = ItemState.FRAUD_CONFIRMED
                        item.fraud_type = FraudType.QUICK_PASS if item.scanner_frames > 0 else (
                            FraudType.CASHIER_SKIP if item.owner_role == 'cashier' else FraudType.MISSED_SCAN
                        )
                        # Use touched_frame if available, otherwise current frame
                        item.fraud_frame = item.touched_frame if item.touched_frame > 0 else frame_idx
                        self.logged_fraud_ids.add(track_id)
                        if item.fraud_type == FraudType.QUICK_PASS:
                            self.quick_pass_events += 1
                        elif item.fraud_type == FraudType.CASHIER_SKIP:
                            self.cashier_skip_events += 1
                        else:
                            self.missed_scans += 1
                        
                        event = {
                            'type': item.fraud_type.value,
                            'track_id': track_id,
                            'label': label,
                            'confidence': score,
                            'frame': frame_idx,
                            'frame_num': frame_idx,
                            'start_frame': item.touched_frame if item.touched_frame > 0 else frame_idx,  # Evidence from this frame
                            'bbox': item.last_bbox,
                            'scanner_time': item.scanner_frames / self.fps,
                            'roi_zone': item.last_known_zone,
                            'roi_confirmed': item.entered_checkout,
                            'owner_id': item.owner_id,
                            'owner_role': item.owner_role,
                            'ownership_history': item.ownership_history,
                            'message': (
                                f"FRAUD CONFIRMED: '{label}' quick-passed scanner ({item.scanner_frames/self.fps:.2f}s)"
                                if item.fraud_type == FraudType.QUICK_PASS else
                                f"FRAUD CONFIRMED: CASHIER skipped scan for '{label}'"
                                if item.fraud_type == FraudType.CASHIER_SKIP else
                                f"FRAUD CONFIRMED: '{label}' left counter unscanned and moving to exit"
                            )
                        }
                        events.append(event)
                        self.fraud_events.append(event)
                        print(f"[FRAUD] ❌ FRAUD CONFIRMED ({item.fraud_type.value}): Item {track_id} '{label}' - moving to exit")

            # ========== PACKING AREA TRACKING (for all items) ==========
            # Track items in packing area regardless of state
            if 'packing_area' in zones:
                if item.packing_area_entry_frame == 0:
                    item.packing_area_entry_frame = frame_idx
                item.packing_area_frames += 1
                
                # Cashier skip shortcut: cashier-owned item in packing area without scan
                if (not item.is_scanned and item.owner_role == 'cashier' and
                    item.scanner_frames < self.min_scanner_frames and
                    track_id not in self.logged_fraud_ids):
                    item.state = ItemState.FRAUD_CONFIRMED
                    item.fraud_type = FraudType.CASHIER_SKIP
                    item.fraud_frame = frame_idx
                    self.logged_fraud_ids.add(track_id)
                    self.cashier_skip_events += 1
                    
                    event = {
                        'type': item.fraud_type.value,
                        'track_id': track_id,
                        'label': label,
                        'confidence': score,
                        'frame': frame_idx,
                        'frame_num': frame_idx,
                        'bbox': item.last_bbox,
                        'scanner_time': item.scanner_frames / self.fps,
                        'time_in_packing': item.packing_area_frames / self.fps,
                        'roi_zone': item.last_known_zone,
                        'roi_confirmed': item.entered_checkout,
                        'owner_id': item.owner_id,
                        'owner_role': item.owner_role,
                        'ownership_history': item.ownership_history,
                        'message': f"FRAUD: CASHIER skipped scan for '{label}' (in packing area)"
                    }
                    events.append(event)
                    self.fraud_events.append(event)
                    print(f"[FRAUD] ❌ FRAUD (cashier_skip): Item {track_id} '{label}' in packing area without scan")
                    continue  # Already frauded; skip warning logic
                
                # Check for WARNING: Item in packing area too long without scan
                if (not item.is_scanned and 
                    not item.packing_area_warning_sent and
                    item.packing_area_frames >= self.packing_area_warning_frames and
                    track_id not in self.logged_warning_ids):
                    
                    item.packing_area_warning_sent = True
                    self.logged_warning_ids.add(track_id)
                    self.packing_area_warnings += 1
                    
                    warning = {
                        'type': 'packing_area_warning',
                        'track_id': track_id,
                        'label': label,
                        'confidence': score,
                        'frame': frame_idx,
                        'frame_num': frame_idx,
                        'bbox': item.last_bbox,
                        'time_in_packing': item.packing_area_frames / self.fps,
                        'roi_zone': item.last_known_zone,
                        'roi_confirmed': item.entered_checkout,
                        'owner_id': item.owner_id,
                        'owner_role': item.owner_role,
                        'ownership_history': item.ownership_history,
                        'message': f"⚠️ WARNING: '{label}' in packing area for {item.packing_area_frames/self.fps:.1f}s WITHOUT SCAN"
                    }
                    self.warning_events.append(warning)
                    events.append(warning)
                    print(f"[WARNING] ⚠️ Item {track_id} '{label}' in PACKING AREA {item.packing_area_frames/self.fps:.1f}s - NOT SCANNED!")
            else:
                # Item left packing area
                if item.packing_area_frames > 0 and not item.is_scanned:
                    # FRAUD: Item left packing area without being scanned
                    if track_id not in self.logged_fraud_ids:
                        item.state = ItemState.FRAUD_CONFIRMED
                        item.fraud_type = FraudType.PACKING_AREA_UNSCANNED
                        item.fraud_frame = frame_idx
                        self.logged_fraud_ids.add(track_id)
                        self.packing_area_frauds += 1
                        
                        event = {
                            'type': 'packing_area_unscanned',
                            'track_id': track_id,
                            'label': label,
                            'confidence': score,
                            'frame': frame_idx,
                            'frame_num': frame_idx,
                            'bbox': item.last_bbox,
                            'time_in_packing': item.packing_area_frames / self.fps,
                            'roi_zone': item.last_known_zone,
                            'roi_confirmed': item.entered_checkout,
                            'owner_id': item.owner_id,
                            'owner_role': item.owner_role,
                            'ownership_history': item.ownership_history,
                            'message': f"❌ FRAUD: '{label}' LEFT packing area - NOT SCANNED (was there {item.packing_area_frames/self.fps:.1f}s)"
                        }
                        events.append(event)
                        self.fraud_events.append(event)
                        print(f"[FRAUD] ❌ Item {track_id} '{label}' LEFT PACKING AREA - NOT SCANNED!")
        
        return events
    
    def _check_fraud_confirmations(self, frame_idx: int) -> List[Dict]:
        """
        Check for fraud confirmations
        
        STRICT RULES - Only trigger fraud when:
        1. Item is in EXIT zone OR moving toward exit
        2. Item has NO scan record
        """
        events = []
        
        for track_id, item in self.items.items():
            # Skip already processed or scanned items
            if item.state in [ItemState.STATIONARY, ItemState.SCANNED, ItemState.FRAUD_CONFIRMED, ItemState.CONCEALED]:
                continue
            
            # Ignore items that never entered the checkout ROI
            if not item.entered_checkout:
                continue
            
            # Skip if already logged
            if track_id in self.logged_fraud_ids:
                continue
            
            # Skip if item is scanned
            if item.is_scanned:
                continue
            
            # Check if item is in exit zone or moving toward exit
            in_exit = False
            moving_to_exit = False
            
            if item.positions:
                last_pos = item.positions[-1]
                in_exit = self._point_in_zone(last_pos, 'exit')
                moving_to_exit = self._is_moving_toward_exit(item)
            
            # FRAUD: Item in exit zone OR moving toward exit without being scanned
            if in_exit or moving_to_exit:
                # For timeout-based: only trigger if item left counter some time ago
                if item.state == ItemState.LEFT_COUNTER:
                    frames_since_exit = frame_idx - item.counter_exit_frame
                    if frames_since_exit < self.fps * 2:  # Wait at least 2 seconds
                        continue
                
                item.state = ItemState.FRAUD_CONFIRMED
                if item.scanner_frames > 0 and item.scanner_frames < self.min_scanner_frames:
                    item.fraud_type = FraudType.QUICK_PASS
                    self.quick_pass_events += 1
                elif item.owner_role == 'cashier':
                    item.fraud_type = FraudType.CASHIER_SKIP
                    self.cashier_skip_events += 1
                else:
                    item.fraud_type = FraudType.MISSED_SCAN
                    self.missed_scans += 1
                item.fraud_frame = frame_idx
                self.logged_fraud_ids.add(track_id)
                
                location = "in EXIT zone" if in_exit else "moving toward exit"
                event = {
                    'type': item.fraud_type.value,
                    'track_id': track_id,
                    'label': item.label,
                    'confidence': item.confidence,
                    'frame': frame_idx,
                    'frame_num': frame_idx,
                    'bbox': item.last_bbox,
                    'scanner_time': item.scanner_frames / self.fps,
                    'roi_zone': item.last_known_zone,
                    'roi_confirmed': item.entered_checkout,
                    'owner_id': item.owner_id,
                    'owner_role': item.owner_role,
                    'ownership_history': item.ownership_history,
                    'message': (
                        f"FRAUD: '{item.label}' QUICK PASS {location} (scanner {item.scanner_frames/self.fps:.2f}s)"
                        if item.fraud_type == FraudType.QUICK_PASS else
                        f"FRAUD: CASHIER skipped scan for '{item.label}' {location}"
                        if item.fraud_type == FraudType.CASHIER_SKIP else
                        f"FRAUD: '{item.label}' {location} - NOT SCANNED"
                    )
                }
                events.append(event)
                self.fraud_events.append(event)
                print(f"[FRAUD] ❌ FRAUD ({item.fraud_type.value}): Item {track_id} '{item.label}' {location}")
        
        return events
    
    def get_evidence_frames(self, fraud_frame: int, before_seconds: float = 3.0, after_frames: int = 75) -> List[np.ndarray]:
        """
        Get evidence frames around a fraud event
        
        Args:
            fraud_frame: Frame number where fraud was detected
            before_seconds: Seconds of video before fraud
            after_frames: Frames to capture after fraud
        
        Returns:
            List of frames for evidence
        """
        frames = []
        
        # Get frames from buffer (before fraud)
        for item in self.frame_buffer:
            if item['frame_num'] <= fraud_frame:
                frames.append(item['frame'])
        
        return frames
    
    def get_summary(self) -> Dict:
        """Get detection summary"""
        return {
            'total_items_tracked': len(self.items),
            'items_on_counter': len(self.counter_items),
            'items_scanned': len(self.scanned_item_ids),
            'missed_scans': self.missed_scans,
            'quick_pass': self.quick_pass_events,
            'cashier_skip': self.cashier_skip_events,
            'packing_area_warnings': self.packing_area_warnings,
            'packing_area_frauds': self.packing_area_frauds,
            'pocket_concealment': self.pocket_concealment,
            'fraud_events': len(self.fraud_events),
            'warning_events': len(self.warning_events),
            'grace_period_active': self.is_in_grace_period,
            'current_frame': self.current_frame
        }


# Detection classes for retail items
RETAIL_DETECTION_CLASSES = [
    "person", "hand", "shopping cart", "shopping basket", "trolley", "cart",
    "bottle", "box", "package", "bag", "can", "container", "carton",
    "food", "drink", "snack", "chips", "cereal", "milk", "juice", "water",
    "product", "item", "goods", "merchandise",
    "grocery", "retail item", "packaged food", "beverage"
]
