#!/usr/bin/env python3
"""
Zone Calibration Tool
Helps you draw and configure detection zones for your specific camera view
Works in headless mode (no display required)
"""
import cv2
import numpy as np
import json
import os

# Disable Qt GUI for headless environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class ZoneCalibrator:
    """Interactive zone calibration tool"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.zones = {}
        self.current_zone = None
        self.current_points = []
        self.frame = None
        self.window_name = "Zone Calibration - Press 'h' for help"
        
        # Zone colors
        self.colors = {
            'counter': (0, 255, 0),      # Green
            'scanner': (255, 0, 0),       # Blue
            'pos': (0, 255, 255),         # Yellow
            'customer_area': (255, 165, 0), # Orange
            'trolley': (128, 0, 128),     # Purple
            'baby_seat': (255, 0, 255),   # Magenta
            'cart_bottom': (0, 128, 128), # Teal
            'exit': (128, 128, 128),      # Gray
            'basket': (0, 200, 0),        # Light green
        }
        
        # Zone descriptions
        self.zone_descriptions = {
            'counter': "Main checkout counter where items are placed for scanning",
            'scanner': "Barcode scanner area - where items pass over scanner",
            'pos': "POS/Register area - cashier side",
            'customer_area': "Customer standing area with cart/trolley",
            'trolley': "Shopping trolley/cart area",
            'baby_seat': "Baby seat area on top of trolley",
            'cart_bottom': "Bottom rack of shopping cart",
            'exit': "Exit area after checkout",
            'basket': "Items in customer's basket/cart before checkout"
        }
    
    def load_frame(self):
        """Load first frame from video"""
        cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"[ERROR] Could not read video: {self.video_path}")
            return False
        
        print(f"[INFO] Loaded frame: {self.frame.shape[1]}x{self.frame.shape[0]}")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_zone:
                self.current_points.append([x, y])
                print(f"  Point {len(self.current_points)}: ({x}, {y})")
    
    def draw_zones(self, frame):
        """Draw all zones on frame"""
        display = frame.copy()
        
        # Draw completed zones
        for zone_name, points in self.zones.items():
            color = self.colors.get(zone_name, (255, 255, 255))
            pts = np.array(points, np.int32)
            cv2.polylines(display, [pts], True, color, 2)
            cv2.fillPoly(display, [pts], (*color[:3], 50))  # Semi-transparent fill
            
            # Label
            if len(points) > 0:
                cv2.putText(display, zone_name, (points[0][0], points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current zone being drawn
        if self.current_zone and len(self.current_points) > 0:
            color = self.colors.get(self.current_zone, (255, 255, 255))
            pts = np.array(self.current_points, np.int32)
            
            # Draw lines between points
            for i in range(len(self.current_points) - 1):
                cv2.line(display, tuple(self.current_points[i]), 
                        tuple(self.current_points[i+1]), color, 2)
            
            # Draw points
            for pt in self.current_points:
                cv2.circle(display, tuple(pt), 5, color, -1)
            
            # Show current zone name
            cv2.putText(display, f"Drawing: {self.current_zone}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return display
    
    def show_help(self):
        """Print help instructions"""
        print("\n" + "="*60)
        print("ZONE CALIBRATION HELP")
        print("="*60)
        print("\nKEYS:")
        print("  1-9: Start drawing zone (see list below)")
        print("  Enter: Finish current zone")
        print("  Backspace: Remove last point")
        print("  c: Clear current zone")
        print("  r: Reset all zones")
        print("  s: Save zones to JSON")
        print("  l: Load zones from JSON")
        print("  q: Quit")
        print("  h: Show this help")
        print("\nZONES:")
        for i, (name, desc) in enumerate(self.zone_descriptions.items(), 1):
            if i <= 9:
                print(f"  {i}: {name} - {desc}")
        print("="*60 + "\n")
    
    def run(self):
        """Run calibration interface"""
        if not self.load_frame():
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.show_help()
        
        zone_keys = list(self.zone_descriptions.keys())
        
        while True:
            display = self.draw_zones(self.frame)
            
            # Show instructions
            cv2.putText(display, "Press 'h' for help, 'q' to quit", 
                       (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            # Number keys for zone selection
            if ord('1') <= key <= ord('9'):
                zone_idx = key - ord('1')
                if zone_idx < len(zone_keys):
                    self.current_zone = zone_keys[zone_idx]
                    self.current_points = []
                    print(f"\n[ZONE] Drawing: {self.current_zone}")
                    print(f"       {self.zone_descriptions[self.current_zone]}")
                    print("       Click points, press Enter when done")
            
            # Enter - finish zone
            elif key == 13:  # Enter
                if self.current_zone and len(self.current_points) >= 3:
                    self.zones[self.current_zone] = self.current_points.copy()
                    print(f"[DONE] Zone '{self.current_zone}' saved with {len(self.current_points)} points")
                    self.current_zone = None
                    self.current_points = []
            
            # Backspace - remove last point
            elif key == 8:  # Backspace
                if self.current_points:
                    self.current_points.pop()
                    print(f"  Removed point, {len(self.current_points)} remaining")
            
            # c - clear current
            elif key == ord('c'):
                self.current_zone = None
                self.current_points = []
                print("[CLEAR] Current zone cleared")
            
            # r - reset all
            elif key == ord('r'):
                self.zones = {}
                self.current_zone = None
                self.current_points = []
                print("[RESET] All zones cleared")
            
            # s - save
            elif key == ord('s'):
                self.save_zones()
            
            # l - load
            elif key == ord('l'):
                self.load_zones()
            
            # h - help
            elif key == ord('h'):
                self.show_help()
            
            # q - quit
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def save_zones(self, filename: str = "zones_config.json"):
        """Save zones to JSON file"""
        output = {
            'video': os.path.basename(self.video_path),
            'frame_size': [self.frame.shape[1], self.frame.shape[0]],
            'zones': {k: v for k, v in self.zones.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[SAVED] Zones saved to {filename}")
        
        # Also print Python code
        print("\n[CODE] Copy this to your script:\n")
        print("# Zone configuration")
        for name, points in self.zones.items():
            print(f"{name.upper()}_ZONE = np.array({points})")
    
    def load_zones(self, filename: str = "zones_config.json"):
        """Load zones from JSON file"""
        if not os.path.exists(filename):
            print(f"[ERROR] File not found: {filename}")
            return
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.zones = data.get('zones', {})
        print(f"[LOADED] {len(self.zones)} zones from {filename}")
    
    def manual_entry(self):
        """Manual zone entry via command line"""
        print("\n" + "="*60)
        print("MANUAL ZONE ENTRY")
        print("="*60)
        print("Enter coordinates in ANY of these formats:")
        print("  Format 1: 300,200 900,200 900,400 300,400")
        print("  Format 2: [[300,200], [900,200], [900,400], [300,400]]")
        print("Press Enter with no input to skip a zone")
        print("Type 'done' when finished\n")
        
        for zone_name, desc in self.zone_descriptions.items():
            print(f"\n[{zone_name}] {desc}")
            coords = input(f"  Enter points for {zone_name}: ").strip()
            
            if coords.lower() == 'done':
                break
            
            if not coords:
                print(f"  Skipped {zone_name}")
                continue
            
            try:
                points = []
                
                # Try JSON format first: [[x,y], [x,y], ...]
                if coords.startswith('['):
                    import json
                    points = json.loads(coords)
                else:
                    # Space-separated format: x1,y1 x2,y2 ...
                    for pair in coords.split():
                        x, y = map(int, pair.split(','))
                        points.append([x, y])
                
                if len(points) >= 3:
                    self.zones[zone_name] = points
                    print(f"  ✓ Added {zone_name} with {len(points)} points")
                else:
                    print(f"  ✗ Need at least 3 points")
            except Exception as e:
                print(f"  ✗ Invalid format: {e}")
        
        if self.zones:
            self.save_zones()
            
            # Also generate Python code
            print("\n[CODE] Add this to your script:\n")
            print("# Zone configuration from calibrator")
            print("ZONES = {")
            for name, points in self.zones.items():
                print(f"    '{name}': np.array({points}),")
            print("}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python zone_calibrator.py <video_path> [--extract]")
        print("\nOptions:")
        print("  --extract    Extract first frame as image for manual zone drawing")
        print("  --manual     Enter zones manually via command line")
        print("\nThis tool helps you calibrate detection zones for your camera view.")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    # Check for headless mode options
    if "--extract" in sys.argv or "--manual" in sys.argv:
        # Headless mode - extract frame and guide manual input
        calibrator = ZoneCalibrator(video_path)
        if calibrator.load_frame():
            # Save frame for reference
            frame_path = "zone_reference_frame.jpg"
            cv2.imwrite(frame_path, calibrator.frame)
            print(f"\n[SAVED] Reference frame saved to: {frame_path}")
            print(f"[INFO] Frame size: {calibrator.frame.shape[1]}x{calibrator.frame.shape[0]}")
            
            print("\n" + "="*60)
            print("MANUAL ZONE CONFIGURATION")
            print("="*60)
            print("\n1. Open 'zone_reference_frame.jpg' in an image viewer")
            print("2. Note the coordinates of each zone corner")
            print("3. Create zones_config.json with format:")
            print('''
{
  "zones": {
    "counter": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "scanner": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "trolley": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "baby_seat": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "cart_bottom": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  }
}
''')
            print("\nZONE DESCRIPTIONS:")
            for name, desc in calibrator.zone_descriptions.items():
                print(f"  {name}: {desc}")
            
            # Offer interactive manual entry
            if "--manual" in sys.argv:
                calibrator.manual_entry()
    else:
        # Try GUI mode (may fail in headless)
        try:
            calibrator = ZoneCalibrator(video_path)
            calibrator.run()
        except Exception as e:
            print(f"[ERROR] GUI mode failed: {e}")
            print("\nRun with --extract for headless mode:")
            print(f"  python zone_calibrator.py {video_path} --extract")


if __name__ == "__main__":
    main()
