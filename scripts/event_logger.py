"""
Event Logger and Reporter
"""
import json
from datetime import datetime
from typing import List, Dict

class EventLogger:
    """Logs and reports fraud/miss events"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.events: List[Dict] = []
    
    def log_event(self, event: Dict):
        """Log a single event"""
        self.events.append(event)
        
        # Print to console
        timestamp = event.get('timestamp', datetime.now()).strftime('%H:%M:%S') if isinstance(event.get('timestamp'), datetime) else str(event.get('timestamp', ''))
        event_type = event.get('type', 'unknown')
        track_id = event.get('track_id', 'N/A')
        label = event.get('label', 'unknown')
        confidence = event.get('confidence', 0)
        print(f"[{timestamp}] {event_type} | ID:{track_id} | {label} ({confidence:.2f})")
        
        # Optionally write to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {event_type} | ID:{track_id} | {label} ({confidence:.2f})\n")
    
    def export_json(self, events: List[Dict], filepath: str):
        """Export events as JSON"""
        data = []
        for e in events:
            # Convert details dict to ensure all values are JSON serializable
            details_clean = {}
            if isinstance(e.get('details'), dict):
                for k, v in e['details'].items():
                    if hasattr(v, 'item'):  # numpy array/type
                        details_clean[k] = float(v) if isinstance(v, (int, float)) else str(v)
                    elif isinstance(v, (int, float)):
                        details_clean[k] = float(v)
                    else:
                        details_clean[k] = v
            
            # Handle timestamp
            timestamp = e.get('timestamp')
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            data.append({
                'timestamp': timestamp_str,
                'event': e.get('type', 'unknown'),
                'tracker_id': int(e.get('track_id', 0)),
                'item': str(e.get('label', 'unknown')),
                'confidence': float(e.get('confidence', 0)),
                'details': details_clean
            })
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[LOG] Exported {len(events)} events to {filepath}")
    
    def print_summary(self, report: Dict):
        """Print summary report"""
        print("\n" + "="*60)
        print("FRAUD DETECTION SUMMARY REPORT")
        print("="*60)
        print(f"Total Events Detected: {report['total_events']}")
        print(f"  - Fraud Events: {report['fraud_count']}")
        print(f"  - Missed Scan Events: {report['miss_count']}")
        print(f"Total Items Tracked: {report['total_items_tracked']}")
        
        if report['fraud_events']:
            print("\n--- FRAUD EVENTS ---")
            for event in report['fraud_events']:
                print(f"  {event}")
        
        if report['miss_events']:
            print("\n--- MISSED SCAN EVENTS ---")
            for event in report['miss_events']:
                print(f"  {event}")
        
        print("="*60 + "\n")
