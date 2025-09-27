
import cv2
import os
import sys
import base64
import json
import numpy as np
import time
import math
import config
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque
from enum import Enum


# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.rabbitmq_client import RabbitMQClient
from detection.config import *


# Object Tracker Class (from your backend)
class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, class_name, confidence):
        self.objects[self.next_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'class_name': class_name,
            'confidence': confidence,
            'trail': deque(maxlen=10),
            'active': True
        }
        self.objects[self.next_id]['trail'].append(centroid)
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        # Mark all objects as inactive initially
        for obj in self.objects.values():
            obj['active'] = False
            
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for detection in detections:
                centroid, bbox, class_name, confidence = detection
                self.register(centroid, bbox, class_name, confidence)
        else:
            object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            detection_centroids = np.array([det[0] for det in detections])
            
            distances = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
            object_ids = list(self.objects.keys())
            used_detection_indices = set()
            
            for i, object_id in enumerate(object_ids):
                if len(used_detection_indices) >= len(detections):
                    break
                
                min_distance = float('inf')
                min_detection_idx = -1
                
                for j, detection in enumerate(detections):
                    if j in used_detection_indices:
                        continue
                    
                    distance = distances[i][j]
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        min_detection_idx = j
                
                if min_detection_idx != -1:
                    centroid, bbox, class_name, confidence = detections[min_detection_idx]
                    self.objects[object_id]['centroid'] = centroid
                    self.objects[object_id]['bbox'] = bbox
                    self.objects[object_id]['class_name'] = class_name
                    self.objects[object_id]['confidence'] = confidence
                    self.objects[object_id]['trail'].append(centroid)
                    self.objects[object_id]['active'] = True
                    self.disappeared[object_id] = 0
                    used_detection_indices.add(min_detection_idx)
                else:
                    self.disappeared[object_id] += 1
            
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    centroid, bbox, class_name, confidence = detection
                    self.register(centroid, bbox, class_name, confidence)
            
            for object_id in list(self.disappeared.keys()):
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects


# ROI State Enum (from your backend)
class ROIState(Enum):
    ROI_2_WITH_SCOOPER = "ROI-2 with scooper"
    ROI_2_WITHOUT_SCOOPER = "ROI-2 without scooper"
    ROI_1 = "ROI-1"


# State-based Violation Detector (from your backend)
class StateBasedHygieneViolationDetector:
    def __init__(self):
        """
        __init__
            Benefit: Initializes violation detection system with state machine and timing controls
            Input: None
            Output: StateBasedHygieneViolationDetector instance
            Purpose: Sets up violation tracking, state history, and timing mechanisms for hygiene protocol enforcement
        """

        self.violations = []
        self.violation_start_timestamp = None
        self.violation_active = False
        self.potential_violation_start = None
        self.stabilization_period = 1.0  # 1 second stabilization period
        
        # State machine components
        self.state_history = deque(maxlen=5)
        self.current_state = None
        self.state_start_timestamp = None
        
        # ROI-1 transition delay mechanism - NOW TIMESTAMP BASED
        self.roi1_disappear_timestamp = None
        self.roi1_to_roi2_delay = 0.6  # 0.5 second delay
        self.last_roi1_state_timestamp = None
        # to solve disappearing from roi1 you need to train yolo model to overfitt on this part so any hand with any shape can detect it 
        # so in this case the roi1_to_roi2_delay will not be important 
        
        # ROI-2 state transition delay mechanism - NOW TIMESTAMP BASED
        self.roi2_scooper_disappear_timestamp = None
        self.roi2_with_to_without_delay = 1.0 # this value must to be more than or equal to self.stabilization_period
        self.last_roi2_with_scooper_timestamp = None
        
    def _determine_current_state(self, roi1_hands, roi2_hands, roi1_scoopers, roi2_scoopers, current_timestamp):

        """
        _determine_current_state
            Benefit: Analyzes current object positions to determine system state (ROI-1, ROI-2 with/without scooper)
            Input: roi1_hands, roi2_hands, roi1_scoopers, roi2_scoopers (dicts), current_timestamp (float)
            Output: ROIState enum value representing current system state
            Purpose: Core state machine logic that determines hand/scooper interaction state with transition delays
        """

        roi1_hand_count = sum(1 for hand in roi1_hands.values() if hand['active'])
        roi2_hand_count = sum(1 for hand in roi2_hands.values() if hand['active'])
        roi1_scooper_count = sum(1 for scooper in roi1_scoopers.values() if scooper['active'])
        roi2_scooper_count = sum(1 for scooper in roi2_scoopers.values() if scooper['active'])
        
        # Handle ROI-1 to ROI-2 transition delay
        if roi1_hand_count > 0:
            self.roi1_disappear_timestamp = None
            self.last_roi1_state_timestamp= current_timestamp
            self.roi2_scooper_disappear_timestamp = None
            self.last_roi2_with_scooper_timestamp = None
            return ROIState.ROI_1
        else:
            if self.last_roi1_state_timestamp is not None:
                if self.roi1_disappear_timestamp is None:
                    self.roi1_disappear_timestamp = current_timestamp
                
                elapsed_since_disappear = current_timestamp - self.roi1_disappear_timestamp
                if elapsed_since_disappear < self.roi1_to_roi2_delay:
                    return ROIState.ROI_1
        
        # Check ROI-2
        if roi2_hand_count > 0:
            self.roi1_disappear_timestamp = None
            self.last_roi1_state_timestamp = None
            
            if roi2_scooper_count > 0:
                self.roi2_scooper_disappear_timestamp = None
                self.last_roi2_with_scooper_timestamp = current_timestamp
                return ROIState.ROI_2_WITH_SCOOPER
            else:
                if self.last_roi2_with_scooper_timestamp is not None:
                    if self.roi2_scooper_disappear_timestamp is None:
                        self.roi2_scooper_disappear_timestamp = current_timestamp
                    
                    elapsed_since_scooper_disappear = current_timestamp - self.roi2_scooper_disappear_timestamp
                    if elapsed_since_scooper_disappear < self.roi2_with_to_without_delay:
                        return ROIState.ROI_2_WITH_SCOOPER
                
                return ROIState.ROI_2_WITHOUT_SCOOPER
        
        return None
    
    def _update_state_history(self, new_state, current_timestamp):
        """
        _update_state_history
            Benefit: Maintains historical record of state changes for pattern analysis
            Input: new_state (ROIState enum), current_timestamp (float)
            Output: Boolean (True if state changed, False if same state)
            Purpose: Tracks state transitions to enable violation pattern detection
        """

        if new_state is None:
            return False
            
        if not self.state_history or self.state_history[-1] != new_state:
            self.state_history.append(new_state)
            self.current_state = new_state
            self.state_start_timestamp = current_timestamp
            print(f"State changed to: {new_state.value}")
            return True
        return False
    
    def _analyze_extended_pattern(self):
        """
        _analyze_extended_pattern
        Benefit: Analyzes complex 4+ state sequences to distinguish legitimate hygiene behavior from violations
        Input: None (uses internal state_history deque)
        Output: Tuple (is_violation: bool, reason: string)
        Purpose: Fallback pattern analyzer that prevents false positives by recognizing legitimate scooper usage patterns in extended sequences
            """
        if len(self.state_history) < 4:
            return False, "Extended pattern analysis inconclusive - insufficient history"
        
        # Look at longer patterns (4+ states)
        recent_states = list(self.state_history)[-4:]
        
        # Check for patterns like: ROI-2_WITHOUT → ROI-1 → ROI-2_WITH → ROI-1 → ROI-2_WITHOUT
        # This could indicate legitimate scooper return followed by another cycle
        
        # Pattern: Previous was ROI-2 without scooper, went to ROI-1, then ROI-2 with scooper, then ROI-1, now ROI-2 without
        if (len(recent_states) >= 4 and 
            recent_states[-1] == ROIState.ROI_2_WITHOUT_SCOOPER and
            recent_states[-2] == ROIState.ROI_1 and
            recent_states[-3] == ROIState.ROI_2_WITH_SCOOPER and
            recent_states[-4] == ROIState.ROI_1):
            
            # This suggests legitimate scooper usage in the middle
            return False, "LEGITIMATE: Extended pattern shows scooper was used in recent cycle"
        
        # Pattern: ROI-2_WITHOUT → ROI-2_WITH → ROI-1 → ROI-2_WITHOUT
        # This suggests hand picked up scooper then went to ROI-1 then back without scooper (legitimate)
        if (len(recent_states) >= 4 and
            recent_states[-1] == ROIState.ROI_2_WITHOUT_SCOOPER and
            recent_states[-2] == ROIState.ROI_1 and
            recent_states[-3] == ROIState.ROI_2_WITH_SCOOPER and
            recent_states[-4] == ROIState.ROI_2_WITHOUT_SCOOPER):
            
            return False, "LEGITIMATE: Hand picked up scooper before ROI-1 transition"
        
        # Default: if we can't determine a clear legitimate pattern, it's potentially a violation
        return True, "POTENTIAL VIOLATION: Extended pattern analysis suggests violation"

    def _analyze_violation_pattern(self):
        """
        _analyze_violation_pattern
            Benefit: Examines state history to detect hygiene violation patterns
            Input: None (uses internal state_history)
            Output: Tuple (violation_detected: bool, reason: string)
            Purpose: Identifies specific violation patterns like ROI-2 access without proper scooper usage
        """

        if len(self.state_history) < 3:
            return False, "Insufficient state history"
        
        recent_states = list(self.state_history)[-3:]
        
        if recent_states[-1] != ROIState.ROI_2_WITHOUT_SCOOPER:
            return False, "Not in ROI-2 without scooper state"
        
        if recent_states[-2] != ROIState.ROI_1:
            return False, "No ROI-1 transition detected"
        
        previous_roi2_state = recent_states[-3]
        
        if previous_roi2_state == ROIState.ROI_2_WITHOUT_SCOOPER:
            return True, "VIOLATION DETECTED"
        elif previous_roi2_state == ROIState.ROI_2_WITH_SCOOPER:
            return False, "LEGITIMATE: Hand returned scooper and came back to ROI-2"
        else:
            return self._analyze_extended_pattern()
        
    def check_violation(self, roi1_hands, roi2_hands, roi1_scoopers, roi2_scoopers, current_timestamp):
            
        """
        check_violation
            Benefit: Main violation detection method that combines all detection logic
            Input: roi1_hands, roi2_hands, roi1_scoopers, roi2_scoopers (dicts), current_timestamp (float)
            Output: Tuple with violation status, object counts, stabilization info, and violation details
            Purpose: Comprehensive violation analysis with stabilization period and detailed reporting
        """

        
        # Count active objects
        roi1_hand_count = sum(1 for hand in roi1_hands.values() if hand['active'])
        roi2_hand_count = sum(1 for hand in roi2_hands.values() if hand['active'])
        roi1_scooper_count = sum(1 for scooper in roi1_scoopers.values() if scooper['active'])
        roi2_scooper_count = sum(1 for scooper in roi2_scoopers.values() if scooper['active'])
        
        # Determine current state with new logic (NOW TIMESTAMP BASED)
        new_state = self._determine_current_state(roi1_hands, roi2_hands, roi1_scoopers, roi2_scoopers, current_timestamp)
        state_changed = self._update_state_history(new_state, current_timestamp)
        
        # Check for potential violation using state machine logic
        state_violation_detected = False
        violation_reason = "No violation"
        
        # Only analyze when we detect hands in ROI-2 without scooper
        if new_state == ROIState.ROI_2_WITHOUT_SCOOPER:
            state_violation_detected, violation_reason = self._analyze_violation_pattern()
        
        # Fallback to original logic if state machine is inconclusive
        original_potential_violation = (
            roi2_hand_count > 0 and  # Hand(s) detected in ROI-2
            roi1_hand_count == 0 and  # No hands in ROI-1
            roi1_scooper_count >= 3 and  # At least 3 scoopers in ROI-1          
            roi2_scooper_count == 0   # No scoopers in ROI-2
        )
        
        # Combine state machine logic with original logic
        if state_violation_detected:
            potential_violation = True  # TRUST THE STATE MACHINE
            violation_type = "STATE-BASED"
        else:
            # If state machine says it's legitimate, be more lenient
            if violation_reason.startswith("LEGITIMATE"):
                potential_violation = False
                violation_type = "STATE-OVERRIDE"
            else:
                potential_violation = original_potential_violation
                violation_type = "ORIGINAL"
        
        # Handle stabilization period (NOW TIMESTAMP BASED)
        if potential_violation:
            if self.potential_violation_start is None:
                self.potential_violation_start = current_timestamp
                print(f"🟡 POTENTIAL VIOLATION STARTED - stabilization timer begins")
                violation_detected = False
            else:
                stabilization_time = current_timestamp - self.potential_violation_start
                print(f"🟡 STABILIZATION: {stabilization_time:.1f}s / {self.stabilization_period}s")
                if stabilization_time >= self.stabilization_period:
                    violation_detected = True
                    print(f"🔴 VIOLATION CONFIRMED after {stabilization_time:.1f}s stabilization")
                else:
                    violation_detected = False
        else:
            if self.potential_violation_start is not None:
                print(f"🟢 POTENTIAL VIOLATION ENDED - timer reset")
            self.potential_violation_start = None
            violation_detected = False
            
        if potential_violation and self.potential_violation_start is not None:
            stabilization_time = current_timestamp - self.potential_violation_start
            print(f"🟡 STABILIZATION DEBUG: {stabilization_time:.1f}s / {self.stabilization_period}s")
            print(f"   Conditions stable: ROI-2 hands={roi2_hand_count}, ROI-1 hands={roi1_hand_count}")
            print(f"   ROI-1 scoopers={roi1_scooper_count}, ROI-2 scoopers={roi2_scooper_count}")
        
        # Handle violation state changes (NOW TIMESTAMP BASED)
        new_violation = None

        # ADD DEBUG CODE HERE:
        print(f"DEBUG VIOLATION COUNTING:")
        print(f"  violation_detected: {violation_detected}")
        print(f"  violation_active: {self.violation_active}")
        print(f"  potential_violation: {potential_violation}")
        print(f"  Can count new violation: {violation_detected and not self.violation_active}")
        print(f"  Current total violations: {len(self.violations)}")

        if violation_detected and not self.violation_active:
            # New violation confirmed after stabilization
            self.violation_active = True
            self.violation_start_timestamp = current_timestamp
            violation_info = {
                'timestamp': float(current_timestamp),  # Use video timestamp
                'roi1_hands': int(roi1_hand_count),  # Convert to Python int
                'roi2_hands': int(roi2_hand_count),
                'roi1_scoopers': int(roi1_scooper_count),
                'roi2_scoopers': int(roi2_scooper_count),
                'violation_type': str(violation_type),  # Ensure string
                'state_reason': str(violation_reason),
                'current_state': str(new_state.value) if new_state else "No State",
                'state_history': [str(state.value) for state in list(self.state_history)[-5:]],
                'message': f"VIOLATION ({violation_type}): {roi2_hand_count} hand(s) in ROI-2, no hands in ROI-1, {roi1_scooper_count} scoopers in ROI-1, no scoopers in ROI-2"
            }
            self.violations.append(violation_info)
            
            # Create new_violation object for immediate notification - ENSURE JSON SERIALIZABLE
            new_violation = {
                'timestamp': float(current_timestamp),  # Video timestamp, not processing time
                'violation_type': str(violation_type),  # Python string
                'message': str('Hand hygiene protocol violated')  # Python string
            }
            
            print(f"VIOLATION: {violation_info['message']}")
            print(f"   Reason: {violation_reason}")
            print(f"   State pattern: {' → '.join(violation_info['state_history'])}")
            
        elif not potential_violation and self.violation_active:
            # Violation resolved (NOW TIMESTAMP BASED)
            self.violation_active = False
            duration = current_timestamp - self.violation_start_timestamp if self.violation_start_timestamp else 0
            print(f"Violation resolved after {duration:.1f} seconds")

        # Calculate remaining stabilization time for display (NOW TIMESTAMP BASED)
        stabilization_remaining = 0
        if self.potential_violation_start is not None and not violation_detected:
            elapsed = current_timestamp - self.potential_violation_start
            stabilization_remaining = max(0, self.stabilization_period - elapsed)
            
        # Determine display state (handle None case)
        display_state = new_state.value if new_state else (self.current_state.value if self.current_state else "No hands detected")
        
        return (violation_detected, roi1_hand_count, roi2_hand_count, 
                roi1_scooper_count, roi2_scooper_count, float(stabilization_remaining),  # Ensure float
                str(display_state), str(violation_reason), new_violation)  # Ensure strings
    
    def reset_for_new_video(self):
        
        """
        reset_for_new_video
            Benefit: Clears all violation detection state for processing new video
            Input: None
            Output: None (resets internal state)
            Purpose: Ensures clean state transition between different video processing sessions
        """
        print("🔄 RESETTING VIOLATION DETECTOR FOR NEW VIDEO")
        
        # Clear all violations
        self.violations = []
        self.violation_start_timestamp = None
        self.violation_active = False
        self.potential_violation_start = None
        self.stabilization_period = 1.0
        
        # Reset state machine
        self.state_history = deque(maxlen=5)
        self.current_state = None
        self.state_start_timestamp = None
        
        # Reset ROI-1 transition delay mechanism
        self.roi1_disappear_timestamp = None
        self.roi1_to_roi2_delay = 0.6
        self.last_roi1_state_timestamp = None
        
        # Reset ROI-2 state transition delay mechanism
        self.roi2_scooper_disappear_timestamp = None
        self.roi2_with_to_without_delay = 1.0
        self.last_roi2_with_scooper_timestamp = None
        
        print(f"✅ Violation detector reset complete - violations count: {len(self.violations)}")
    

# Enhanced Detection Service
class HygieneDetectionService:
    def __init__(self):
        """
        __init__
            Benefit: Initializes complete detection service with all components
            Input: None
            Output: HygieneDetectionService instance
            Purpose: Sets up RabbitMQ client, YOLO model, trackers, violation detector, and ROI configuration
        """
        self.rabbitmq_client = RabbitMQClient(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            username=RABBITMQ_USERNAME,
            password=RABBITMQ_PASSWORD
        )
        self.model = None
        self.setup_directories()
        self.load_yolo_model()
        
        # Initialize trackers
        self.roi1_hand_tracker = ObjectTracker(max_disappeared=90, max_distance=200)
        self.roi1_scooper_tracker = ObjectTracker(max_disappeared=70, max_distance=200)
        self.roi2_hand_tracker = ObjectTracker(max_disappeared=90, max_distance=200)
        self.roi2_scooper_tracker = ObjectTracker(max_disappeared=70, max_distance=200)
        
        # Initialize violation detector
        self.violation_detector = StateBasedHygieneViolationDetector()
        
        # ROI configuration
        self.roi1_points = None
        self.roi2_points = None
        
        # Color palettes (from your backend)
        self.roi1_hand_colors = config.ROI1_HAND_COLORS
        self.roi1_scooper_colors = config.ROI1_SCOOPER_COLORS
        self.roi2_hand_colors = config.ROI2_HAND_COLORS
        self.roi2_scooper_colors = config.ROI2_SCOOPER_COLORS
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def load_yolo_model(self):
        """Load YOLO model for hands and scoopers"""
        print("Loading YOLO model for hygiene monitoring...")
        
        model_path = config.YOLO_MODEL_PATH

        # Check if model file exists before attempting to load
        if not os.path.exists(model_path):
            error_msg = f"YOLO model doesn't exist at '{model_path}'"
            print(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        try:
            self.model = YOLO(model_path)
            print(f"YOLO hygiene model loaded successfully from: {model_path}")
        except Exception as e:
            error_msg = f"Error loading YOLO model from '{model_path}': {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    def base64_to_frame(self, base64_string):
        """Convert base64 string back to OpenCV frame"""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"Error converting base64 to frame: {e}")
            return None
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return None
    
    def setup_roi_from_config(self, roi_config):
        """Setup ROI points from configuration - ENHANCED"""
        if roi_config and 'roi1' in roi_config and 'roi2' in roi_config:
            self.roi1_points = np.array(roi_config['roi1'], dtype=np.int32)
            self.roi2_points = np.array(roi_config['roi2'], dtype=np.int32)
            config_label = roi_config.get('label', 'Custom Configuration')
            print(f"ROI configured: {config_label}")
            print(f"ROI-1 points: {roi_config['roi1']}")
            print(f"ROI-2 points: {roi_config['roi2']}")
            
            # Store the current configuration label for comparison
            self.current_roi_label = config_label
            return True
        else:
            # Default ROI points (fallback)
            self.roi1_points = np.array([[468, 267], [378, 678], [469, 703], [546, 274]], dtype=np.int32)
            self.roi2_points = np.array([[470, 704], [546, 274], [639, 282], [610, 723]], dtype=np.int32)
            self.current_roi_label = "Default Configuration"
            print("Using default ROI configuration")
            return True
    
    def reset_for_new_video(self):
        """Reset detection service state for new video processing"""
        print("🔄 RESETTING DETECTION SERVICE FOR NEW VIDEO...")
        
        # Reset ROI configuration
        self.roi1_points = None
        self.roi2_points = None
        self.current_roi_label = None
        
        # Reset trackers
        self.roi1_hand_tracker = ObjectTracker(max_disappeared=90, max_distance=200)
        self.roi1_scooper_tracker = ObjectTracker(max_disappeared=70, max_distance=200)
        self.roi2_hand_tracker = ObjectTracker(max_disappeared=90, max_distance=200)
        self.roi2_scooper_tracker = ObjectTracker(max_disappeared=70, max_distance=200)
        
        # IMPORTANT: Reset violation detector completely
        self.violation_detector.reset_for_new_video()
        
        # Reset hand state history if it exists
        if hasattr(self, 'hand_state_history'):
            self.hand_state_history = {}
            print("✅ Hand state history cleared")
        
        print("✅ Detection service reset complete")
    
    def is_point_in_roi(self, point, roi_polygon):
        """Check if point is inside ROI polygon"""
        point_tuple = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(roi_polygon, point_tuple, False) >= 0
    
    def bbox_intersects_roi(self, bbox, roi_polygon):
        """Check if bounding box center is in ROI"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return self.is_point_in_roi(center, roi_polygon)
    
    def get_object_roi(self, bbox):
        """Determine which ROI an object is in"""
        if self.bbox_intersects_roi(bbox, self.roi1_points):
            return 'ROI1'
        elif self.bbox_intersects_roi(bbox, self.roi2_points):
            return 'ROI2'
        return None
    
    def detect_objects(self, frame):
        """Detect hands and scoopers in frame"""
        try:
            results = self.model(frame, conf=0.25)
            target_classes = ['scooper', 'hand']
            
            # Prepare detections for tracking
            roi1_hand_detections = []
            roi1_scooper_detections = []
            roi2_hand_detections = []
            roi2_scooper_detections = []
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    if class_name in target_classes and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        object_roi = self.get_object_roi((x1, y1, x2, y2))
                        
                        if object_roi == 'ROI1':
                            if class_name == 'hand':
                                roi1_hand_detections.append((centroid, (x1, y1, x2, y2), class_name, confidence))
                            elif class_name == 'scooper':
                                roi1_scooper_detections.append((centroid, (x1, y1, x2, y2), class_name, confidence))
                        elif object_roi == 'ROI2':
                            if class_name == 'hand':
                                roi2_hand_detections.append((centroid, (x1, y1, x2, y2), class_name, confidence))
                            elif class_name == 'scooper':
                                roi2_scooper_detections.append((centroid, (x1, y1, x2, y2), class_name, confidence))
            
            return roi1_hand_detections, roi1_scooper_detections, roi2_hand_detections, roi2_scooper_detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], [], [], []
    
    def draw_roi(self, frame, roi_polygon, roi_label, color):
        """Draw ROI on frame"""
        cv2.polylines(frame, [roi_polygon], True, color, 2)
        for i, point in enumerate(roi_polygon):
            cv2.circle(frame, tuple(point), 5, color, -1)
            cv2.putText(frame, f'{roi_label}-P{i+1}', (point[0] + 10, point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], color)
        frame = cv2.addWeighted(frame, 0.95, overlay, 0.05, 0)
        return frame
    
    def draw_tracked_objects(self, frame, tracked_objects, colors, roi_prefix, violation_mode=False):
        """
        draw_tracked_objects
            Benefit: Renders tracked objects with trails, bounding boxes, and labels
            Input: frame, tracked_objects (dict), colors (list), roi_prefix (string), violation_mode (bool)
            Output: None (modifies frame in place)
            Purpose: Visual representation of tracking state and object movement patterns
        """
        for obj_id, obj in tracked_objects.items():
            if not obj['active']:
                continue
                
            x1, y1, x2, y2 = obj['bbox']
            centroid = obj['centroid']
            
            if violation_mode and self.violation_detector.violation_active:
                color = (0, 0, 255)  # Red for violation
            else:
                color = colors[obj_id % len(colors)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if violation_mode else 2)
            
            # Draw trail
            if len(obj['trail']) > 1:
                for i in range(1, len(obj['trail'])):
                    cv2.line(frame, obj['trail'][i-1], obj['trail'][i], color, 2)
                    cv2.circle(frame, obj['trail'][i], 3, color, -1)
            
            cv2.circle(frame, centroid, 4, color, -1)
            
            # Draw label
            label_text = f"{roi_prefix}-{obj['class_name'][0].upper()}{obj_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
            
            cv2.rectangle(frame, 
                        (x1, y1 - text_size[1] - 10), 
                        (x1 + text_size[0] + 10, y1), 
                        (0, 0, 0), -1)
            cv2.rectangle(frame, 
                        (x1, y1 - text_size[1] - 10), 
                        (x1 + text_size[0] + 10, y1), 
                        color, 2)
            cv2.putText(frame, label_text, 
                      (x1 + 5, y1 - 5), 
                      font, font_scale, (255, 255, 255), 1)
    
    def draw_violation_status(self, frame, violation_detected, roi1_hand_count, roi2_hand_count, 
                             roi1_scooper_count, roi2_scooper_count, stabilization_remaining, 
                             current_state, violation_reason):
        """Draw violation status on frame"""
        # Status background
        status_height = 240
        cv2.rectangle(frame, (10, 10), (750, status_height), (0, 0, 0), -1)
        
        # Violation status
        if violation_detected:
            status_color = (0, 0, 255)  # Red
            status_text = "VIOLATION DETECTED"
        elif stabilization_remaining > 0:
            status_color = (0, 165, 255)  # Orange
            status_text = f"CHECKING... {stabilization_remaining:.1f}s"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "COMPLIANT"
        
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Object counts
        cv2.putText(frame, f"ROI-1 Hands: {roi1_hand_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ROI-2 Hands: {roi2_hand_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ROI-1 Scoopers: {roi1_scooper_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ROI-2 Scoopers: {roi2_scooper_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # State information
        cv2.putText(frame, f"Current State: {current_state}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Violation reason
        reason_text = violation_reason if len(violation_reason) < 60 else violation_reason[:57] + "..."
        cv2.putText(frame, f"Analysis: {reason_text}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Violation count
        violation_count = len(self.violation_detector.violations)
        cv2.putText(frame, f"Total Violations: {violation_count}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def convert_to_json_serializable(self, obj):
        """
        convert_to_json_serializable
            Benefit: Ensures all data types can be properly encoded as JSON for message transmission
            Input: obj (any Python object with potential numpy types)
            Output: JSON-serializable equivalent object
            Purpose: Prevents serialization errors when sending results through message queue
        """

        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (deque,)):
            return [self.convert_to_json_serializable(item) for item in list(obj)]
        else:
            return obj
    
    def _serialize_tracking_data(self, tracked_objects):
        """Convert tracking data to JSON-serializable format - UPDATED"""
        """
        _serialize_tracking_data
            Benefit: Converts tracking data structures to JSON-compatible format
            Input: tracked_objects (dictionary)
            Output: Serialized dictionary with converted data types
            Purpose: Enables transmission of tracking information to downstream services
        """
        serialized = {}
        for obj_id, obj in tracked_objects.items():
            serialized[str(obj_id)] = {  # Convert key to string
                'centroid': [int(obj['centroid'][0]), int(obj['centroid'][1])],  # Convert to int
                'bbox': [int(x) for x in obj['bbox']],  # Convert bbox coordinates to int
                'class_name': str(obj['class_name']),
                'confidence': float(obj['confidence']),  # Ensure it's Python float
                'active': bool(obj['active']),
                'trail': [[int(point[0]), int(point[1])] for point in list(obj['trail'])]  # Convert trail points
            }
        return serialized
    
    def draw_frame_annotations(self, frame, roi1_hands, roi1_scoopers, roi2_hands, roi2_scoopers,
                              violation_detected, roi1_hand_count, roi2_hand_count,
                              roi1_scooper_count, roi2_scooper_count, stabilization_remaining,
                              current_state, violation_reason):
        """Draw all annotations on the frame"""
        annotated_frame = frame.copy()
        
        # Draw ROIs
        annotated_frame = self.draw_roi(annotated_frame, self.roi1_points, "ROI1", (0, 255, 255))  # Yellow
        annotated_frame = self.draw_roi(annotated_frame, self.roi2_points, "ROI2", (255, 0, 255))  # Magenta
        
        # Draw tracked objects
        self.draw_tracked_objects(annotated_frame, roi1_hands, self.roi1_hand_colors, "R1")
        self.draw_tracked_objects(annotated_frame, roi1_scoopers, self.roi1_scooper_colors, "R1")
        self.draw_tracked_objects(annotated_frame, roi2_hands, self.roi2_hand_colors, "R2", violation_mode=True)
        self.draw_tracked_objects(annotated_frame, roi2_scoopers, self.roi2_scooper_colors, "R2")
        
        # Draw violation status
        self.draw_violation_status(annotated_frame, violation_detected, roi1_hand_count, roi2_hand_count,
                                 roi1_scooper_count, roi2_scooper_count, stabilization_remaining,
                                 current_state, violation_reason)
        
        return annotated_frame
    
    def debug_detection_state(self, roi1_hands, roi1_scoopers, roi2_hands, roi2_scoopers, 
                         roi1_hand_detections, roi1_scooper_detections, 
                         roi2_hand_detections, roi2_scooper_detections):
        """Debug method to print detailed detection information"""
        
        print(f"\n🔍 DEBUG DETECTION STATE:")
        print(f"Raw detections:")
        print(f"  ROI-1: {len(roi1_hand_detections)} hands, {len(roi1_scooper_detections)} scoopers")
        print(f"  ROI-2: {len(roi2_hand_detections)} hands, {len(roi2_scooper_detections)} scoopers")
        
        print(f"Active tracked objects:")
        roi1_hand_count = sum(1 for hand in roi1_hands.values() if hand['active'])
        roi2_hand_count = sum(1 for hand in roi2_hands.values() if hand['active'])
        roi1_scooper_count = sum(1 for scooper in roi1_scoopers.values() if scooper['active'])
        roi2_scooper_count = sum(1 for scooper in roi2_scoopers.values() if scooper['active'])
        
        print(f"  ROI-1: {roi1_hand_count} hands, {roi1_scooper_count} scoopers")
        print(f"  ROI-2: {roi2_hand_count} hands, {roi2_scooper_count} scoopers")
        
        # Check violation conditions
        violation_condition = (
            roi2_hand_count > 0 and 
            roi1_hand_count == 0 and 
            roi1_scooper_count >= 3 and 
            roi2_scooper_count == 0
        )
        
        print(f"Violation condition check:")
        print(f"  ROI-2 has hands: {roi2_hand_count > 0} ({roi2_hand_count} hands)")
        print(f"  ROI-1 no hands: {roi1_hand_count == 0} ({roi1_hand_count} hands)")
        print(f"  ROI-1 ≥3 scoopers: {roi1_scooper_count >= 3} ({roi1_scooper_count} scoopers)")
        print(f"  ROI-2 no scoopers: {roi2_scooper_count == 0} ({roi2_scooper_count} scoopers)")
        print(f"  → Should violate: {violation_condition}")
        
        # Show state history
        if hasattr(self.violation_detector, 'state_history') and self.violation_detector.state_history:
            recent_states = [state.value for state in list(self.violation_detector.state_history)[-3:]]
            print(f"Recent state history: {' → '.join(recent_states)}")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def start_consuming(self):
        """Start consuming messages from the raw frames queue"""
        """
        start_consuming
            Benefit: Initiates RabbitMQ message consumption loop
            Input: None
            Output: Boolean (False if connection fails)
            Purpose: Main service entry point that begins processing incoming frame messages
        """

        if not self.rabbitmq_client.connect():
            print("Failed to connect to RabbitMQ")
            return False
        
        print("Starting hygiene detection service...")
        print("Waiting for frames to process...")
        
        try:
            # Set up consumer
            self.rabbitmq_client.channel.basic_consume(
                queue=RAW_FRAMES_QUEUE,
                on_message_callback=self.process_frame_message,
                auto_ack=True
            )
            
            # Start consuming
            self.rabbitmq_client.channel.start_consuming()
            
        except KeyboardInterrupt:
            print("Service interrupted by user")
            self.rabbitmq_client.channel.stop_consuming()
        except Exception as e:
            print(f"Error in service: {e}")
        finally:
            self.rabbitmq_client.close()
    
    def process_frame_message(self, channel, method, properties, body):
        """Process incoming frame message from RabbitMQ - WITH VIDEO RESET"""
        """
        process_frame_message
            Benefit: Main message processing pipeline that handles incoming frames from RabbitMQ
            Input: RabbitMQ message parameters (channel, method, properties, body)
            Output: None (publishes results to detection results queue)
            Purpose: Core service loop that processes frames, detects violations, and forwards results
        """
        try:
            """ message from frame reader service ()
            message = {
                            'frame_id': f'frame_{processed_count}_{int(time.time() * 1000)}',
                            'timestamp': datetime.now().isoformat(),
                            'frame_data': frame_base64,
                            'frame_number': processed_count,  # Use processed count for frame numbering
                            'original_frame_number': frame_count,  # Keep original for reference
                            'video_path': video_path,
                            'frame_width': frame.shape[1],
                            'frame_height': frame.shape[0],
                            'total_frames': total_frames,
                            'fps': fps
                        }
            
            """
            message = json.loads(body)
            
            # Handle new video start signal - ADD THIS CHECK
            if message.get('new_video_start', False):
                print("🆕 NEW VIDEO START SIGNAL RECEIVED")
                self.reset_for_new_video()
                return
            
            # Handle end-of-video signals
            if message.get('end_of_video', False):
                print("Received end-of-video signal, forwarding...")
                clean_message = self.convert_to_json_serializable(message)
                self.rabbitmq_client.publish_message(DETECTION_RESULTS_QUEUE, clean_message)
                return
            
            # Rest of the existing method remains the same...
            frame_id = message.get('frame_id')
            frame_data = message.get('frame_data')
            timestamp = message.get('timestamp')
            frame_number = message.get('frame_number')
            roi_config = message.get('roi_config')
            
            # Check if this is frame 1 of a new video (additional safety check)
            if frame_number == 1 and not hasattr(self, '_current_video_processed'):
                print("🔍 Detected frame #1 - ensuring clean state")
                # This ensures we're starting fresh even if new_video_start wasn't sent
                if len(self.violation_detector.violations) > 0:
                    print("⚠️  Found existing violations, resetting...")
                    self.reset_for_new_video()
                self._current_video_processed = True
            
            # FIXED: Always setup ROI if provided (not just when None)
            if roi_config:
                current_config_label = roi_config.get('label', 'Unknown')
                
                # Check if this is a new configuration
                if (self.roi1_points is None or self.roi2_points is None or 
                    not hasattr(self, 'current_roi_label') or 
                    self.current_roi_label != current_config_label):
                    
                    print(f"Setting up new ROI configuration: {current_config_label}")
                    self.setup_roi_from_config(roi_config)
                    self.current_roi_label = current_config_label
                
            # Skip processing if no ROI configured
            if self.roi1_points is None or self.roi2_points is None:
                print("No ROI configuration available, skipping frame")
                return
            
            print(f"Processing hygiene frame {frame_id} (#{frame_number})")
            
            # Convert base64 to frame
            frame = self.base64_to_frame(frame_data)
            if frame is None:
                print(f"Failed to decode frame {frame_id}")
                return
            
            ############################################################################
            # Detect objects
            roi1_hand_detections, roi1_scooper_detections, roi2_hand_detections, roi2_scooper_detections = self.detect_objects(frame)
            
            # Update trackers
            roi1_tracked_hands = self.roi1_hand_tracker.update(roi1_hand_detections)
            roi1_tracked_scoopers = self.roi1_scooper_tracker.update(roi1_scooper_detections)
            roi2_tracked_hands = self.roi2_hand_tracker.update(roi2_hand_detections)
            roi2_tracked_scoopers = self.roi2_scooper_tracker.update(roi2_scooper_detections)
            
            # ADD DEBUG OUTPUT EVERY 10th FRAME
            if frame_number % 10 == 0:  # Debug every 10th frame to avoid spam
                self.debug_detection_state(
                    roi1_tracked_hands, roi1_tracked_scoopers, 
                    roi2_tracked_hands, roi2_tracked_scoopers,
                    roi1_hand_detections, roi1_scooper_detections,
                    roi2_hand_detections, roi2_scooper_detections
                )
            
            # Calculate video timestamp from frame data
            frame_number = message.get('frame_number', 1)
            fps = message.get('fps', 30)
            video_timestamp = frame_number / fps  # This is the key fix!

            # Check for violations with enhanced state-based logic
            (violation_detected, roi1_hand_count, roi2_hand_count, roi1_scooper_count, 
            roi2_scooper_count, stabilization_remaining, current_state, violation_reason,
            new_violation) = self.violation_detector.check_violation(
                roi1_tracked_hands, roi1_tracked_scoopers, roi2_tracked_hands, roi2_tracked_scoopers, video_timestamp
            )
            
            # Draw ROIs and tracked objects on frame
            processed_frame = self.draw_frame_annotations(
                frame, roi1_tracked_hands, roi1_tracked_scoopers, 
                roi2_tracked_hands, roi2_tracked_scoopers,
                violation_detected, roi1_hand_count, roi2_hand_count,
                roi1_scooper_count, roi2_scooper_count, stabilization_remaining,
                current_state, violation_reason
            )
            
            # Convert processed frame to base64
            processed_frame_base64 = self.frame_to_base64(processed_frame)
            if processed_frame_base64 is None:
                print(f"Failed to encode processed frame {frame_id}")
                return
            
            # Prepare enhanced detection result message
            result_message = {
                'frame_id': str(frame_id),
                'timestamp': str(timestamp),
                'processed_frame': str(processed_frame_base64),
                'frame_number': int(frame_number),
                'hygiene_data': {
                    'violation_detected': bool(violation_detected),
                    'roi1_hand_count': int(roi1_hand_count),
                    'roi2_hand_count': int(roi2_hand_count),
                    'roi1_scooper_count': int(roi1_scooper_count),
                    'roi2_scooper_count': int(roi2_scooper_count),
                    'current_state': str(current_state),
                    'violation_reason': str(violation_reason),
                    'stabilization_remaining': float(stabilization_remaining),
                    'total_violations': int(len(self.violation_detector.violations)),
                    'new_violation': new_violation
                },
                'tracking_data': {
                    'roi1_hands': self._serialize_tracking_data(roi1_tracked_hands),
                    'roi2_hands': self._serialize_tracking_data(roi2_tracked_hands),
                    'roi1_scoopers': self._serialize_tracking_data(roi1_tracked_scoopers),
                    'roi2_scoopers': self._serialize_tracking_data(roi2_tracked_scoopers)
                }
            }
            
            # DEBUG: Log violation information
            current_violation_count = len(self.violation_detector.violations)
            if violation_detected or new_violation or current_violation_count > 0:
                print(f"DETECTION SERVICE DEBUG - Frame {frame_number}:")
                print(f"  Current violation detected: {violation_detected}")
                print(f"  New violation: {new_violation is not None}")
                print(f"  Total violations in detector: {current_violation_count}")
                print(f"  Violation reason: {violation_reason}")
                print(f"  ROI-2 hands: {roi2_hand_count}, ROI-1 scoopers: {roi1_scooper_count}")
                
            # Final conversion to ensure everything is JSON serializable
            clean_result_message = self.convert_to_json_serializable(result_message)
            
            # DEBUG: Verify the message being sent
            if current_violation_count > 0:
                print(f"SENDING MESSAGE WITH {clean_result_message['hygiene_data']['total_violations']} total violations")
            
            # Send to detection results queue
            if self.rabbitmq_client.publish_message(DETECTION_RESULTS_QUEUE, clean_result_message):
                print(f"✅ Processed hygiene frame {frame_id} - Status: {current_state}")
                if violation_detected:
                    print(f"🚨 VIOLATION: {violation_reason}")
                if current_violation_count > 0:
                    print(f"📊 Total violations so far: {current_violation_count}")
            else:
                print(f"❌ Failed to send processed frame {frame_id}")
                
        except Exception as e:
            print(f"Error processing frame message: {e}")
            import traceback
            traceback.print_exc()
    
    def get_violation_summary(self):
        """Get summary of all violations detected"""
        """
        get_violation_summary
            Benefit: Provides comprehensive summary of all detected violations
            Input: None
            Output: Dictionary with violation statistics and history
            Purpose: Enables reporting and analysis of hygiene compliance over entire video
        """
        return {
            'total_violations': len(self.violation_detector.violations),
            'violations': self.violation_detector.violations,
            'current_state': self.violation_detector.current_state.value if self.violation_detector.current_state else "No State",
            'state_history': [state.value for state in list(self.violation_detector.state_history)]
        }
        
    def get_hand_state_summary(self):
        """
        Get summary of all hand states for analysis and debugging
        """
        if not hasattr(self, 'hand_state_history'):
            return {"message": "No hand state history available"}
        
        summary = {
            "total_hands_tracked": len(self.hand_state_history),
            "hands": {}
        }
        
        for hand_id, history in self.hand_state_history.items():
            summary["hands"][hand_id] = {
                "total_states": len(history['states']),
                "current_state": history['states'][-1].value if history['states'] else "Unknown",
                "state_sequence": [state.value for state in history['states']],
                "roi": history['roi'],
                "tracking_duration": history['last_update'] - history['first_seen'],
                "last_seen": history['last_update']
            }
        
        return summary



def main():
    """
    main
        Benefit: Service entry point with error handling and cleanup
        Input: None (command line execution)
        Output: System exit (prints violation summary)
        Purpose: Provides robust service startup with proper exception handling and reporting
    """
    service = HygieneDetectionService()
    
    try:
        service.start_consuming()
    except KeyboardInterrupt:
        print("\nDetection service stopped by user")
    except Exception as e:
        print(f"Detection service error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print violation summary
        summary = service.get_violation_summary()
        print(f"\n=== HYGIENE DETECTION SUMMARY ===")
        print(f"Total violations detected: {summary['total_violations']}")
        print(f"Final state: {summary['current_state']}")
        
        if summary['violations']:
            print("\nViolation Details:")
            for i, violation in enumerate(summary['violations'], 1):
                print(f"{i}. {violation.get('message', 'Unknown violation')} at {violation.get('timestamp', 'Unknown time')}")
                print(f"   Type: {violation.get('violation_type', 'Unknown')}")
                print(f"   Reason: {violation.get('state_reason', 'No reason')}")
                if 'state_history' in violation:
                    print(f"   Pattern: {' → '.join(violation['state_history'])}")

if __name__ == "__main__":
    main()