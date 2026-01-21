"""
POS Data Processor for Retail Fraud Detection
Loads, parses, and matches POS data with detection results
"""
import xml.etree.ElementTree as ET
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import requests
from config_models import POSConfig


class POSProcessor:
    """
    Load and process Point-of-Sale data.
    Matches detected items with POS scanned items to identify fraud.
    """
    
    def __init__(self, config: POSConfig, fps: int = 25, video_start_time: Optional[datetime] = None):
        """
        Args:
            config: POSConfig with XML path, data format, matching strategy
            fps: Frames per second of video
            video_start_time: Start time of video (for time synchronization)
        """
        self.config = config
        self.fps = fps
        self.video_start_time = video_start_time or datetime.now()
        self.pos_items = []
        self.item_names_set = set()  # For quick lookup
        
        if config.enabled and config.xml_path:
            self.load_pos_data()
    
    def load_pos_data(self) -> List[Dict]:
        """
        Load POS data from configured source (XML, CSV, or API).
        
        Returns:
            List of POS items with format: {'time': datetime, 'item': str, 'raw_data': dict}
        """
        if self.config.data_format.value == 'xml':
            return self._load_xml()
        elif self.config.data_format.value == 'csv':
            return self._load_csv()
        elif self.config.data_format.value == 'api':
            return self._load_api()
        else:
            raise ValueError(f"Unknown data format: {self.config.data_format}")
    
    def _load_xml(self) -> List[Dict]:
        """Parse POS XML file"""
        if not self.config.xml_path or not self.config.xml_path.endswith('.xml'):
            print(f"[POS] XML path not configured or invalid: {self.config.xml_path}")
            return []
        
        try:
            tree = ET.parse(self.config.xml_path)
            root = tree.getroot()
            sales = []
            
            print(f"\n{'='*60}")
            print(f"📋 POS XML DATA LOADED: {self.config.xml_path}")
            print(f"{'='*60}")
            
            # Find all ART_SALE elements (each represents a scanned item)
            for sale in root.findall('.//ART_SALE'):
                # Get timestamp from header
                hdr = sale.find('.//Hdr')
                raw_date_elem = hdr.find('.//szTaCreatedDate') if hdr is not None else None
                
                # Get item description from ARTICLE block
                article = sale.find('.//ARTICLE')
                item_desc_elem = article.find('.//szDesc1') if article is not None else None
                
                if raw_date_elem is not None and item_desc_elem is not None:
                    raw_date = raw_date_elem.text
                    item_desc = item_desc_elem.text
                    
                    try:
                        dt_obj = datetime.strptime(raw_date, "%Y%m%d%H%M%S")
                        
                        # Apply timezone offset if configured
                        if self.config.timezone_offset != 0:
                            dt_obj += timedelta(hours=self.config.timezone_offset)
                        
                        sales.append({
                            'time': dt_obj,
                            'item': item_desc.strip() if item_desc else '',
                            'raw_time': raw_date,
                            'raw_data': {'type': 'xml', 'sale_elem': sale}
                        })
                        self.item_names_set.add(item_desc.strip().lower())
                        print(f"  POS: {item_desc} @ {dt_obj.strftime('%H:%M:%S')}")
                    except ValueError as e:
                        print(f"  [WARN] Could not parse date: {raw_date}")
            
            print(f"{'='*60}")
            print(f"Total POS items: {len(sales)}")
            print(f"{'='*60}\n")
            
            self.pos_items = sales
            return sales
        
        except Exception as e:
            print(f"[POS] Error parsing XML: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _load_csv(self) -> List[Dict]:
        """Parse POS CSV file"""
        if not self.config.xml_path or not self.config.xml_path.endswith(('.csv', '.txt')):
            print(f"[POS] CSV path not configured or invalid: {self.config.xml_path}")
            return []
        
        try:
            sales = []
            with open(self.config.xml_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        item = row.get(self.config.item_name_field, '').strip()
                        time_str = row.get(self.config.timestamp_field, '')
                        
                        # Try common datetime formats
                        dt_obj = None
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y%m%d%H%M%S', '%d/%m/%Y %H:%M:%S']:
                            try:
                                dt_obj = datetime.strptime(time_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if dt_obj and item:
                            if self.config.timezone_offset != 0:
                                dt_obj += timedelta(hours=self.config.timezone_offset)
                            
                            sales.append({
                                'time': dt_obj,
                                'item': item,
                                'raw_time': time_str,
                                'raw_data': {'type': 'csv', 'row': row}
                            })
                            self.item_names_set.add(item.lower())
                    except Exception as e:
                        print(f"  [WARN] Error parsing CSV row: {e}")
            
            print(f"[POS] Loaded {len(sales)} items from CSV")
            self.pos_items = sales
            return sales
        
        except Exception as e:
            print(f"[POS] Error loading CSV: {e}")
            return []
    
    def _load_api(self) -> List[Dict]:
        """Fetch POS data from API"""
        if not self.config.api_endpoint:
            print(f"[POS] API endpoint not configured")
            return []
        
        try:
            headers = {}
            if self.config.api_key:
                headers['Authorization'] = f"Bearer {self.config.api_key}"
            
            response = requests.get(self.config.api_endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            sales = []
            
            # Assume response is list of items with 'time' and 'item' fields
            for item_data in data if isinstance(data, list) else data.get('items', []):
                try:
                    item = item_data.get(self.config.item_name_field, '').strip()
                    time_str = item_data.get(self.config.timestamp_field, '')
                    
                    dt_obj = datetime.fromisoformat(time_str)
                    if self.config.timezone_offset != 0:
                        dt_obj += timedelta(hours=self.config.timezone_offset)
                    
                    sales.append({
                        'time': dt_obj,
                        'item': item,
                        'raw_time': time_str,
                        'raw_data': {'type': 'api', 'data': item_data}
                    })
                    self.item_names_set.add(item.lower())
                except Exception as e:
                    print(f"  [WARN] Error parsing API item: {e}")
            
            print(f"[POS] Loaded {len(sales)} items from API")
            self.pos_items = sales
            return sales
        
        except Exception as e:
            print(f"[POS] Error fetching from API: {e}")
            return []
    
    def get_items_by_video_time(self, frame_num: int, time_window_seconds: float = 30.0) -> List[Dict]:
        """
        Get all POS items that should have been scanned up to current video time,
        within a recent time window.
        
        Args:
            frame_num: Current frame number in video
            time_window_seconds: How far back to look for scans (seconds)
            
        Returns:
            List of POS items scanned up to current time
        """
        elapsed_seconds = frame_num / self.fps
        current_video_time = self.video_start_time + timedelta(seconds=elapsed_seconds)
        start_window = current_video_time - timedelta(seconds=time_window_seconds)
        
        return [
            item for item in self.pos_items 
            if start_window <= item['time'] <= current_video_time
        ]
    
    def match_detections_to_pos(
        self,
        frame_num: int,
        detected_labels: List[str]
    ) -> Dict:
        """
        Compare detected items with expected POS items at current frame time.
        
        Args:
            frame_num: Current frame number
            detected_labels: List of labels detected in current frame
            
        Returns:
            Dict with keys:
            - matched_count: Number of detections matched to POS items
            - unmatched_pos_items: Number of POS items not detected
            - extra_detected_items: Number of detections not in POS
            - unmatched_details: List of unmatched POS items
            - fraud_candidates: Potential fraudulent items
        """
        expected_items = self.get_items_by_video_time(frame_num)
        detected_labels_copy = [label.strip().lower() for label in detected_labels]
        
        matched_count = 0
        unmatched_items = []
        fraud_candidates = []
        
        for expected_item in expected_items:
            item_name = expected_item['item'].lower()
            
            # Try to find a match in detected labels
            match_idx = self._find_best_match(item_name, detected_labels_copy)
            
            if match_idx is not None:
                matched_count += 1
                detected_labels_copy.pop(match_idx)
            else:
                unmatched_items.append(expected_item)
                fraud_candidates.append({
                    'type': 'missed_detection',
                    'expected_item': expected_item['item'],
                    'frame_num': frame_num,
                    'timestamp': expected_item['time']
                })
        
        # Remaining detected items are not in POS (potential fraud)
        for detected in detected_labels_copy:
            fraud_candidates.append({
                'type': 'extra_detection',
                'detected_item': detected,
                'frame_num': frame_num,
                'not_in_pos': True
            })
        
        return {
            'matched_count': matched_count,
            'unmatched_pos_items': len(unmatched_items),
            'extra_detected_items': len(detected_labels_copy),
            'unmatched_details': unmatched_items,
            'fraud_candidates': fraud_candidates,
            'total_expected': len(expected_items),
            'total_detected': len(detected_labels)
        }
    
    def _find_best_match(self, pos_item: str, detected_labels: List[str], threshold: Optional[float] = None) -> Optional[int]:
        """
        Find best matching detected item for a POS item.
        
        Args:
            pos_item: POS item name (lowercase)
            detected_labels: List of detected labels (lowercase)
            threshold: Match confidence threshold (uses config if not provided)
            
        Returns:
            Index of best match in detected_labels, or None if no match
        """
        if not detected_labels:
            return None
        
        if threshold is None:
            threshold = self.config.min_match_confidence
        
        strategy = self.config.match_strategy.value
        
        if strategy == 'exact':
            # Exact match only
            for i, label in enumerate(detected_labels):
                if pos_item == label:
                    return i
            return None
        
        elif strategy == 'substring':
            # Substring matching (case-insensitive)
            for i, label in enumerate(detected_labels):
                if pos_item in label or label in pos_item:
                    return i
            return None
        
        elif strategy == 'fuzzy':
            # Fuzzy matching using SequenceMatcher
            best_idx = None
            best_ratio = 0.0
            
            for i, label in enumerate(detected_labels):
                ratio = SequenceMatcher(None, pos_item, label).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
            
            if best_ratio >= threshold:
                return best_idx
            return None
        
        return None
    
    def get_pos_summary(self) -> Dict:
        """Get summary statistics of loaded POS data"""
        return {
            'total_items': len(self.pos_items),
            'unique_items': len(self.item_names_set),
            'date_range': (
                min(item['time'] for item in self.pos_items).isoformat() if self.pos_items else None,
                max(item['time'] for item in self.pos_items).isoformat() if self.pos_items else None
            ),
            'data_format': self.config.data_format.value
        }
