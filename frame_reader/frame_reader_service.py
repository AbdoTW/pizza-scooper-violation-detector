import cv2
import os
import sys
import time
import base64
import uuid
import argparse
import json
from datetime import datetime
import numpy as np

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.rabbitmq_client import RabbitMQClient
from frame_reader.config import *

class FrameReaderService:
    def __init__(self):
        self.rabbitmq_client = RabbitMQClient(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            username=RABBITMQ_USERNAME,
            password=RABBITMQ_PASSWORD
        )
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return None
    
    def send_reset_signal(self):
        """Send reset signal to detection service for new video"""
        try:
            reset_message = {
                'new_video_start': True,
                'message': 'New video processing started - please reset all state',
                'timestamp': datetime.now().isoformat(),
                'frame_id': f'RESET_SIGNAL_{uuid.uuid4()}',
                'source': 'frame_reader_service',
                'reset_trigger': 'new_video_upload'
            }
            
            if self.rabbitmq_client.publish_message(RAW_FRAMES_QUEUE, reset_message):
                print("📤 Reset signal sent to detection service")
                return True
            else:
                print("❌ Failed to send reset signal")
                return False
        except Exception as e:
            print(f"Error sending reset signal: {e}")
            return False
    
    def send_end_of_video_signal(self, video_path, total_processed_frames):
        """Send completion signal to detection results queue"""
        try:
            completion_message = {
                'frame_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'end_of_video': True,
                'video_path': video_path,
                'total_processed_frames': total_processed_frames,
                'processing_complete': True,
                'processed_frame': '',  # Keep empty for completion signal
                'hygiene_data': {       # New format
                    'violation_detected': False,
                    'total_violations': 0,
                    'current_state': 'Processing completed',
                    'violation_reason': 'Video processing finished'
                },
                'frame_number': total_processed_frames
            }
            
            # Send completion signal to detection results queue
            # This ensures the streaming service receives it
            success = self.rabbitmq_client.publish_message('detection_results', completion_message)
            
            if success:
                print(f"✅ Sent end-of-video signal for {video_path}")
            else:
                print(f"❌ Failed to send end-of-video signal")
                
            return success
            
        except Exception as e:
            print(f"Error sending end-of-video signal: {e}")
            return False
    
    def process_video(self, video_path, roi_config=None, send_reset=True):
        """Process video file and extract frames"""
        print(f"Starting to process video: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
        
        # Connect to RabbitMQ
        if not self.rabbitmq_client.connect():
            print("Failed to connect to RabbitMQ")
            return False
        
        # Send reset signal at the start of new video processing
        if send_reset:
            print("🔄 Sending reset signal for new video...")
            self.send_reset_signal()
            time.sleep(0.5)  # Give detection service time to process reset
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties: FPS={fps}, Total Frames={total_frames}, Duration={duration:.2f}s")
        if roi_config:
            print(f"ROI Configuration: {roi_config.get('label', 'Custom')}")
        
        # Calculate frame skip interval
        frame_skip = max(1, int(fps / FRAME_RATE)) if fps > 0 else 1
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on desired frame rate
                if frame_count % frame_skip == 0:
                    
                    # Convert to base64
                    frame_base64 = self.frame_to_base64(frame)
                    
                    if frame_base64:
                        processed_count += 1
                        
                        # Create message
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
                        
                        # Add ROI configuration if provided
                        if roi_config:
                            message['roi_config'] = roi_config
                        
                        # Send to RabbitMQ
                        if self.rabbitmq_client.publish_message(RAW_FRAMES_QUEUE, message):
                            # Progress reporting every 30 frames
                            if processed_count % 30 == 0:
                                elapsed = time.time() - start_time
                                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                                progress = (frame_count / total_frames) * 100
                                print(f"Progress: {processed_count} processed ({progress:.1f}%) - {fps_actual:.1f} fps")
                        else:
                            print(f"Failed to send frame {processed_count}")
                            break
                
                frame_count += 1
                
                # Add small delay to avoid overwhelming the system
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            cap.release()
            
            # CRITICAL: Send end-of-video signal after processing completes
            if processed_count > 0:
                print(f"📹 Video processing completed. Sending completion signal...")
                self.send_end_of_video_signal(video_path, processed_count)
                # Wait a moment to ensure the message is sent
                time.sleep(1)
            
            self.rabbitmq_client.close()
            
            # Final summary
            elapsed_time = time.time() - start_time
            print(f"\n✅ Video processing completed!")
            print(f"   Original frames: {frame_count}")
            print(f"   Processed frames: {processed_count}")
            print(f"   Processing time: {elapsed_time:.2f} seconds")
            print(f"   Average FPS: {processed_count / elapsed_time:.2f}")
        
        return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Frame Reader Service for Hygiene Monitoring')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--roi-config', help='ROI configuration as JSON string')
    parser.add_argument('--new-video', action='store_true', 
                       help='Flag indicating this is a new video (triggers reset)')
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Check if file has supported format
    file_ext = os.path.splitext(args.video_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported video format. Supported formats: {SUPPORTED_FORMATS}")
        sys.exit(1)
    
    # Parse ROI configuration if provided
    roi_config = None
    if args.roi_config:
        try:
            roi_config = json.loads(args.roi_config)
            print(f"ROI configuration loaded: {roi_config.get('label', 'Custom')}")
        except json.JSONDecodeError as e:
            print(f"Error parsing ROI configuration: {e}")
            sys.exit(1)
    
    # Create service and process video
    service = FrameReaderService()
    
    # Process video with reset signal if it's a new video
    success = service.process_video(
        args.video_path, 
        roi_config=roi_config,
        send_reset=args.new_video  # Only send reset if this is flagged as new video
    )
    
    if success:
        print("Frame reader service completed successfully")
        sys.exit(0)
    else:
        print("Frame reader service failed")
        sys.exit(1)

if __name__ == "__main__":
    main()