import os
import json
import threading
import time
import uuid
import subprocess
import cv2
import numpy as np
import base64
from datetime import datetime
from typing import List, Dict, Any
from collections import deque
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import config

import sys
sys.path.append('..')
from shared.rabbitmq_client import RabbitMQClient
from streaming.config import *



class WebSocketManager:
    def __init__(self):
        """
        __init__
            Benefit: Initializes WebSocket connection manager with thread-safe connection tracking
            Input: None
            Output: WebSocketManager instance
            Purpose: Sets up connection list and threading lock for managing multiple concurrent WebSocket clients
        """
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        """
        connect
            Benefit: Accepts and registers new WebSocket connection
            Input: websocket (WebSocket object)
            Output: None (adds connection to active list)
            Purpose: Establishes WebSocket connection and adds client to broadcast list
        """
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """
        disconnect
            Benefit: Safely removes WebSocket connection from active connections
            Input: websocket (WebSocket object)
            Output: None (removes connection from active list)
            Purpose: Cleans up disconnected clients to prevent broadcast errors
        """
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """
        broadcast
            Benefit: Sends message to all active WebSocket connections with error handling
            Input: message (dict)
            Output: None (sends message to all clients)
            Purpose: Distributes real-time updates to all connected frontend clients simultaneously
        """
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        with self.lock:
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)



class HygieneStreamingService:
    def __init__(self):
        """Enhanced initialization with frame tracking for video synchronization"""
        self.websocket_manager = WebSocketManager()
        self.rabbitmq_client = None
        self.frame_buffer = deque(maxlen=100)
        
        # Hygiene monitoring specific stats
        self.stats = {
            "processing_status": "idle",  # idle, processing, completed, error
            "current_video": None,
            "total_frames": 0,
            "violation_detected": False,
            "total_violations": 0,
            "roi1_hand_count": 0,
            "roi2_hand_count": 0,
            "roi1_scooper_count": 0,
            "roi2_scooper_count": 0,
            "current_state": "No hands detected",
            "violation_reason": "No violation detected",
            "stabilization_remaining": 0,
            "start_time": None
        }
        self.stats_lock = threading.Lock()
        self.is_consuming = False
        self.consumer_thread = None
        self.loop = None
        self.broadcast_queue = asyncio.Queue()
        
        # Video processing for saving annotated video
        self.video_writer = None
        self.output_video_path = None
        self.current_roi_config = None
        
        # NEW: Enhanced frame tracking for proper video finalization
        self.expected_total_frames = 0
        self.received_frame_count = 0
        self.last_frame_timestamp = None
        self.finalization_timeout = 30.0  # seconds to wait after last frame
        self.finalization_timer = None
        self.video_finalized = False
        self.frame_numbers_received = set()  # Track which frames we've received
        self.finalization_lock = threading.Lock()  # Thread safety for finalization
        
        # Ensure directories exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
        os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
    
    def setup_rabbitmq(self):
        """Initialize RabbitMQ connection"""
        """
        setup_rabbitmq
        Benefit: Establishes connection to RabbitMQ message queue system
        Input: None
        Output: Boolean (True if connection successful, False otherwise)
        Purpose: Enables communication with detection service through message queue
        """
        try:
            self.rabbitmq_client = RabbitMQClient(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                username=RABBITMQ_USERNAME,
                password=RABBITMQ_PASSWORD
            )
            return self.rabbitmq_client.connect()
        except Exception as e:
            print(f"RabbitMQ setup failed: {e}")
            return False
    
    def cleanup_previous_files(self):     # Clean up streaming/uploads & streaming/processed_videos
        """Clean up previous upload and processed video files"""
        try:
            # Clean uploads directory
            uploads_cleaned = 0
            if os.path.exists(UPLOADS_DIR):
                for filename in os.listdir(UPLOADS_DIR):
                    file_path = os.path.join(UPLOADS_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        uploads_cleaned += 1
            
            # Clean processed videos directory
            processed_cleaned = 0
            if os.path.exists(PROCESSED_VIDEOS_DIR):
                for filename in os.listdir(PROCESSED_VIDEOS_DIR):
                    file_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        processed_cleaned += 1
            
            print(f"🧹 Cleanup complete: {uploads_cleaned} uploads, {processed_cleaned} processed videos removed")
            return True
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return False  
    
    def handle_detection_result(self, ch, method, properties, body):
        """
        handle_detection_result
            Benefit: Processes incoming detection results from RabbitMQ and updates statistics
            Input: RabbitMQ message parameters (ch, method, properties, body)
            Output: None (updates internal stats and broadcasts to WebSocket clients)
            Purpose: Core message processing that converts detection results to WebSocket messages
        """
        try:
            data = json.loads(body.decode('utf-8'))
            current_time = time.time()
            
            # Handle end-of-video signal - DON'T finalize immediately
            if data.get("end_of_video", False) or data.get("processing_complete", False):
                print("📹 End-of-video signal received")
                
                # Store expected total frames for validation
                self.expected_total_frames = data.get("total_processed_frames", self.received_frame_count)
                print(f"Expected total frames: {self.expected_total_frames}")
                print(f"Currently received: {self.received_frame_count}")
                
                # Check if we already have all frames
                if self.received_frame_count >= self.expected_total_frames and self.expected_total_frames > 0:
                    print("All frames already received - finalizing immediately")
                    self._finalize_video_safely()
                else:
                    remaining = self.expected_total_frames - self.received_frame_count
                    print(f"Waiting for remaining {remaining} frames")
                    # Start timeout timer as backup
                    self._start_finalization_timer()
                
                # Send completion signal to frontend (but video not ready yet)
                completion_message = {
                    "type": "processing_complete",
                    "message": "Video processing completed - finalizing video...",
                    "timestamp": datetime.now().isoformat(),
                    "total_frames": self.expected_total_frames,
                    "received_frames": self.received_frame_count,
                    "video_ready": False  # Will be set to True when actually finalized
                }
                
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_queue.put(completion_message), 
                        self.loop
                    )
                
                return  # Don't process as regular frame
            
            # Process regular hygiene detection frame
            frame_number = data.get("frame_number", 0)
            hygiene_data = data.get("hygiene_data", {})
            
            # NEW: Enhanced frame tracking
            if frame_number > 0:
                # Track frame reception with duplicate detection
                if frame_number not in self.frame_numbers_received:
                    self.frame_numbers_received.add(frame_number)
                    old_count = self.received_frame_count
                    self.received_frame_count = len(self.frame_numbers_received)
                    self.last_frame_timestamp = current_time
                    
                    # Only log every 50th frame to reduce spam
                    if self.received_frame_count % 50 == 0 or self.received_frame_count != old_count:
                        print(f"📦 Received frame {frame_number} (total: {self.received_frame_count})")
                    
                    # Check if this completes our expected frame count
                    if (self.expected_total_frames > 0 and 
                        self.received_frame_count >= self.expected_total_frames and 
                        not self.video_finalized):
                        
                        print("✅ All expected frames received - finalizing video")
                        self._cancel_finalization_timer()
                        self._finalize_video_safely()
                else:
                    print(f"⚠️ Duplicate frame {frame_number} received - skipping")
                    return  # Skip duplicate frame processing
            
            # Update hygiene monitoring stats
            with self.stats_lock:
                self.stats["total_frames"] = self.received_frame_count  # Use actual received count
                self.stats["violation_detected"] = hygiene_data.get("violation_detected", False)
                self.stats["roi1_hand_count"] = hygiene_data.get("roi1_hand_count", 0)
                self.stats["roi2_hand_count"] = hygiene_data.get("roi2_hand_count", 0)
                self.stats["roi1_scooper_count"] = hygiene_data.get("roi1_scooper_count", 0)
                self.stats["roi2_scooper_count"] = hygiene_data.get("roi2_scooper_count", 0)
                self.stats["current_state"] = hygiene_data.get("current_state", "No hands detected")
                self.stats["violation_reason"] = hygiene_data.get("violation_reason", "No violation detected")
                self.stats["stabilization_remaining"] = hygiene_data.get("stabilization_remaining", 0)
                
                # Enhanced violation count tracking
                detection_total_violations = hygiene_data.get("total_violations", 0)
                if detection_total_violations > self.stats["total_violations"]:
                    self.stats["total_violations"] = detection_total_violations
                    print(f"VIOLATION COUNT UPDATE: Backend reports {detection_total_violations} total violations")
                
                # Check for new violations in current frame
                if hygiene_data.get("new_violation"):
                    print(f"NEW VIOLATION DETECTED IN FRAME: {hygiene_data.get('new_violation')}")
            
            # Prepare enhanced WebSocket message
            fps = data.get("fps", 30)
            video_timestamp = frame_number / fps if fps > 0 else 0

            websocket_message = {
                "type": "frame_update",
                "frame_data": data.get("processed_frame", ""),
                "frame_number": frame_number,
                "fps": fps,
                "video_timestamp": video_timestamp,
                "hygiene_stats": {
                    "violation_detected": hygiene_data.get("violation_detected", False),
                    "roi1_hand_count": hygiene_data.get("roi1_hand_count", 0),
                    "roi2_hand_count": hygiene_data.get("roi2_hand_count", 0),
                    "roi1_scooper_count": hygiene_data.get("roi1_scooper_count", 0),
                    "roi2_scooper_count": hygiene_data.get("roi2_scooper_count", 0),
                    "current_state": hygiene_data.get("current_state", "No hands detected"),
                    "violation_reason": hygiene_data.get("violation_reason", "No violation detected"),
                    "stabilization_remaining": hygiene_data.get("stabilization_remaining", 0),
                    "new_violation": hygiene_data.get("new_violation"),
                    "total_violations": self.stats["total_violations"],
                    "timing_mode": "video_based",
                    # NEW: Progress tracking
                    "progress": {
                        "received_frames": self.received_frame_count,
                        "expected_frames": self.expected_total_frames,
                        "completion_percentage": (self.received_frame_count / self.expected_total_frames * 100) if self.expected_total_frames > 0 else 0
                    }
                }
            }
                        
            # Save processed frame ONLY if video writer is still active
            if not self.video_finalized:
                self.save_processed_frame(data.get("processed_frame", ""))
            else:
                print(f"Skipping frame {frame_number} - video already finalized")
            
            # Add to buffer
            self.frame_buffer.append(websocket_message)
            
            # Queue message for async broadcasting
            try:
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_queue.put(websocket_message), 
                        self.loop
                    )
            except Exception as e:
                print(f"Failed to queue message for broadcast: {e}")
            
        except Exception as e:
            print(f"Error handling detection result: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _start_finalization_timer(self):
        """Start a timer that will finalize video after timeout"""
        with self.finalization_lock:
            if self.finalization_timer:
                self.finalization_timer.cancel()
            
            def timeout_finalize():
                missing_frames = self.expected_total_frames - self.received_frame_count
                print(f"⏰ Timeout reached after {self.finalization_timeout}s")
                print(f"   Finalizing video with {self.received_frame_count}/{self.expected_total_frames} frames")
                if missing_frames > 0:
                    print(f"   Missing {missing_frames} frames - they may have been lost in processing")
                self._finalize_video_safely()
            
            self.finalization_timer = threading.Timer(self.finalization_timeout, timeout_finalize)
            self.finalization_timer.start()
            print(f"Started finalization timer: {self.finalization_timeout}s")
    
    def _cancel_finalization_timer(self):
        """Cancel the finalization timer"""
        with self.finalization_lock:
            if self.finalization_timer:
                self.finalization_timer.cancel()
                self.finalization_timer = None
                print("Cancelled finalization timer")
    
    def _finalize_video_safely(self):
        """Thread-safe video finalization that can only happen once"""
        with self.finalization_lock:
            if self.video_finalized:
                print("Video already finalized - skipping")
                return
            
            self.video_finalized = True
        
        self._cancel_finalization_timer()
        
        print(f"🎬 Finalizing video with {self.received_frame_count} frames")
        
        # Log detailed frame statistics for debugging
        if self.expected_total_frames > 0:
            expected_frames = set(range(1, self.expected_total_frames + 1))
            missing_frames = expected_frames - self.frame_numbers_received
            completion_rate = (self.received_frame_count / self.expected_total_frames) * 100
            
            print(f"📊 Video Statistics:")
            print(f"   Expected frames: {self.expected_total_frames}")
            print(f"   Received frames: {self.received_frame_count}")
            print(f"   Completion rate: {completion_rate:.1f}%")
            
            if missing_frames:
                missing_count = len(missing_frames)
                if missing_count <= 20:
                    print(f"   Missing frames: {sorted(list(missing_frames))}")
                else:
                    sample_missing = sorted(list(missing_frames))[:20]
                    print(f"   Missing frames (first 20): {sample_missing}...")
                    print(f"   Total missing: {missing_count}")
        
        video_path = self.finalize_processed_video()
        
        if video_path:
            # Update processing status
            with self.stats_lock:
                self.stats["processing_status"] = "completed"
            
            # Notify frontend that video is actually ready for download
            completion_message = {
                "type": "video_ready",
                "message": "Processed video is ready for download",
                "timestamp": datetime.now().isoformat(),
                "video_ready": True,
                "final_frame_count": self.received_frame_count,
                "expected_frame_count": self.expected_total_frames,
                "completion_percentage": (self.received_frame_count / self.expected_total_frames * 100) if self.expected_total_frames > 0 else 100
            }
            
            try:
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_queue.put(completion_message), 
                        self.loop
                    )
                print("✅ Video finalization complete and frontend notified")
            except Exception as e:
                print(f"Failed to broadcast video ready message: {e}")
        else:
            print("❌ Video finalization failed")
            # Notify frontend of failure
            error_message = {
                "type": "video_error",
                "message": "Failed to finalize processed video",
                "timestamp": datetime.now().isoformat(),
                "video_ready": False
            }
            try:
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_queue.put(error_message), 
                        self.loop
                    )
            except:
                pass
    
    def save_processed_frame(self, base64_frame: str):
        """Enhanced frame saving with finalization and duplicate checks"""
        try:
            if not base64_frame or self.video_finalized:
                if self.video_finalized:
                    print("Skipping frame save - video already finalized")
                return
            
            # Decode base64 frame
            frame_data = base64.b64decode(base64_frame)
            frame_array = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Warning: Failed to decode frame for video")
                return
            
            # Write frame only if video writer is active and not finalized
            if (self.video_writer is not None and 
                self.video_writer.isOpened() and 
                not self.video_finalized):
                
                self.video_writer.write(frame)
            else:
                status = []
                if self.video_writer is None:
                    status.append("writer=None")
                elif not self.video_writer.isOpened():
                    status.append("writer=closed")
                if self.video_finalized:
                    status.append("finalized=True")
                print(f"Skipping frame write - {', '.join(status)}")
                
        except Exception as e:
            print(f"Error saving processed frame: {e}")
    
    def initialize_video_writer_early(self, filename: str, width: int, height: int, original_fps: float):
        """Initialize video writer before processing starts"""
        """
        initialize_video_writer
            Benefit: Sets up OpenCV video writer for creating processed video output
            Input: first_frame (numpy array)
            Output: None (initializes self.video_writer)
            Purpose: Configures video encoding parameters and output path for processed video creation
        """
        try:
            # Generate output path
            base_name = os.path.splitext(filename)[0]
            self.output_video_path = os.path.join(
                PROCESSED_VIDEOS_DIR, 
                f"processed_{base_name}.mp4"
            )
            
            # Calculate appropriate output FPS
            # Use original FPS for now - can be adjusted based on frame processing rate if needed
            output_fps = original_fps
            
            # Use H.264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            
            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                output_fps,
                (width, height)
            )
            
            # Verify video writer is opened
            if not self.video_writer.isOpened():
                print("H264 codec failed, trying XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    output_fps,
                    (width, height)
                )
            
            if self.video_writer.isOpened():
                print(f"Video writer initialized EARLY: {self.output_video_path}")
                print(f"Dimensions: {width}x{height}, FPS: {output_fps}")
            else:
                print(f"Failed to initialize video writer early")
                self.video_writer = None
                
        except Exception as e:
            print(f"Error initializing video writer early: {e}")
            self.video_writer = None
    

    
    def finalize_processed_video(self):
        """Finalize and save the processed video"""
        """
        finalize_processed_video
            Benefit: Closes video writer and finalizes processed video file
            Input: None
            Output: String (path to finalized video) or None if failed
            Purpose: Completes video file creation and verifies output file integrity
        """
        try:
            if self.video_writer is not None:
                print("Finalizing processed video...")
                
                # Properly release the video writer
                self.video_writer.release()
                self.video_writer = None
                
                # Verify the file was created and has content
                if self.output_video_path and os.path.exists(self.output_video_path):
                    file_size = os.path.getsize(self.output_video_path)
                    if file_size > 0:
                        print(f"Processed video saved: {self.output_video_path}")
                        print(f"File size: {file_size / 1024 / 1024:.2f} MB")
                        return self.output_video_path
                    else:
                        print(f"Video file is empty: {self.output_video_path}")
                        return None
                else:
                    print(f"Video file not found: {self.output_video_path}")
                    return None
                
        except Exception as e:
            print(f"Error finalizing video: {e}")
            return None
    
    def start_consuming(self):
        """Start consuming messages from RabbitMQ in a separate thread"""
        """
        start_consuming
            Benefit: Initiates RabbitMQ message consumption in separate thread
            Input: None
            Output: None (starts background consumer thread)
            Purpose: Begins listening for detection results from detection service
        """
        if self.is_consuming:
            return
        
        if not self.setup_rabbitmq():
            raise Exception("Failed to setup RabbitMQ connection")
        
        self.is_consuming = True
        self.consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.consumer_thread.start()
    
    def _consume_loop(self):
        """RabbitMQ consumption loop"""
        """
        _consume_loop
            Benefit: Background thread function that continuously consumes RabbitMQ messages
            Input: None
            Output: None (runs until stopped)
            Purpose: Maintains persistent connection to message queue for real-time processing
        """
        try:
            # Use the detection_results queue from your detection service
            self.rabbitmq_client.channel.queue_declare(queue="detection_results", durable=True)
            self.rabbitmq_client.channel.basic_consume(
                queue="detection_results",
                on_message_callback=self.handle_detection_result,
                auto_ack=True
            )
            self.rabbitmq_client.channel.start_consuming()
        except Exception as e:
            print(f"Consumer loop error: {e}")
            self.is_consuming = False
    
    def stop_consuming(self):
        """Enhanced stop consuming with proper video finalization"""
        print("🛑 Stopping consumption and finalizing video...")
        
        if self.rabbitmq_client:
            try:
                self.rabbitmq_client.channel.stop_consuming()
                self.rabbitmq_client.close()
            except:
                pass
        self.is_consuming = False
        
        # Enhanced finalization: ensure video is properly closed
        if not self.video_finalized:
            print("Forcing video finalization due to service shutdown")
            self._finalize_video_safely()
    
    def update_processing_status(self, status: str, video_name: str = None):
        """
        update_processing_status
            Benefit: Updates processing status with thread-safe statistics modification
            Input: status (string), video_name (string, optional)
            Output: None (updates internal stats)
            Purpose: Tracks processing state for frontend status reporting
        """
        with self.stats_lock:
            self.stats["processing_status"] = status
            if video_name:
                self.stats["current_video"] = video_name
            if status == "processing" and not self.stats["start_time"]:
                self.stats["start_time"] = datetime.now().isoformat()
    
    def reset_stats(self):
        """Enhanced reset with comprehensive frame tracking cleanup"""
        with self.stats_lock:
            self.stats = {
                "processing_status": "idle",
                "current_video": None,
                "total_frames": 0,
                "violation_detected": False,
                "total_violations": 0,
                "roi1_hand_count": 0,
                "roi2_hand_count": 0,
                "roi1_scooper_count": 0,
                "roi2_scooper_count": 0,
                "current_state": "No hands detected",
                "violation_reason": "No violation detected",
                "stabilization_remaining": 0,
                "start_time": None
            }
        
        # NEW: Reset enhanced frame tracking
        self.expected_total_frames = 0
        self.received_frame_count = 0
        self.last_frame_timestamp = None
        self.video_finalized = False
        self.frame_numbers_received.clear()
        
        # Cancel any pending finalization timer
        self._cancel_finalization_timer()
        
        # Clean up video processing
        if self.video_writer is not None:
            try:
                if not self.video_finalized:
                    print("Releasing video writer during reset")
                    self.video_writer.release()
            except Exception as e:
                print(f"Error releasing video writer: {e}")
            self.video_writer = None
        
        self.output_video_path = None
        self.current_roi_config = None
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        # Clean up streaming/uploads & streaming/processed_videos
        # self.cleanup_previous_files()
        print("📝 Enhanced stats and video processing state reset complete")
    
    def trigger_frame_reader(self, video_path: str, roi_config: dict = None):
        """
        trigger_frame_reader
            Benefit: Launches frame reader service as subprocess with ROI configuration
            Input: video_path (string), roi_config (dict, optional)
            Output: Boolean (True if successfully started, False otherwise)
            Purpose: Initiates video processing pipeline by starting frame extraction service
        """

        try:
            # Store ROI config for this processing session
            self.current_roi_config = roi_config

            # Create command to run frame reader
            frame_reader_path = os.path.join("..", "frame_reader", "frame_reader_service.py")

            # Prepare command arguments
            cmd = ["python", frame_reader_path, video_path, "--new-video"]

            # If ROI config provided, pass it as JSON string
            if roi_config:
                cmd.extend(["--roi-config", json.dumps(roi_config)])

            # Define log file paths
            stdout_log_path = os.path.join(os.path.dirname(__file__), "frame_reader_out.log")
            stderr_log_path = os.path.join(os.path.dirname(__file__), "frame_reader_err.log")

            # Open log files and start subprocess
            out_log = open(stdout_log_path, "w")
            err_log = open(stderr_log_path, "w")

            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(__file__),
                stdout=out_log,
                stderr=err_log
            )

            print(f"Started frame reader for: {video_path}")
            if roi_config:
                print(f"ROI Config: {roi_config.get('label', 'Custom')}")
            print(f"Logs: {stdout_log_path} (stdout), {stderr_log_path} (stderr)")

            return True

        except Exception as e:
            print(f"Failed to trigger frame reader: {e}")
            return False
    
    async def broadcast_worker(self):
        """
        broadcast_worker
            Benefit: Background async task that handles WebSocket message broadcasting
            Input: None
            Output: None (continuously processes broadcast queue)
            Purpose: Manages asynchronous WebSocket message distribution to connected clients
        """
        while True:
            try:
                message = await self.broadcast_queue.get()
                await self.websocket_manager.broadcast(message)
                self.broadcast_queue.task_done()
            except Exception as e:
                print(f"Broadcast worker error: {e}")


# Global service instance
streaming_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    global streaming_service
    # Startup
    streaming_service.loop = asyncio.get_event_loop()
    streaming_service.start_consuming()
    
    # Start broadcast worker
    broadcast_task = asyncio.create_task(streaming_service.broadcast_worker())
    
    yield
    
    # Shutdown
    broadcast_task.cancel()
    streaming_service.stop_consuming()
    
# FastAPI app setup
app = FastAPI(title="Enhanced Hygiene Monitoring Streaming Service", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced service instance
streaming_service = HygieneStreamingService()

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    upload_video
        Benefit: Handles video file upload with format validation and unique naming
        Input: file (UploadFile)
        Output: JSONResponse with upload status and filename
        Purpose: Receives video files from frontend and prepares them for processing
    """
    try:
        # Validate file format
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Supported: {supported_formats}"
            )
        # Clean up streaming/uploads & streaming/processed_videos
        print(f"🧹 Cleaning up previous files before new upload...")
        streaming_service.cleanup_previous_files()
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reset stats for new video
        streaming_service.reset_stats()
        streaming_service.update_processing_status("uploaded", unique_filename)
        
        return JSONResponse({
            "success": True,
            "filename": unique_filename,
            "originalName": file.filename,
            "message": "Video uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/start-stream")
async def start_stream(data: dict):
    """Start video processing pipeline"""
    try:
        filename = data.get("filename")
        roi_config = data.get("roi_config")
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Get video properties BEFORE starting processing
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Extract video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Update status
        streaming_service.update_processing_status("processing", filename)
        
        # Initialize video writer BEFORE starting frame reader
        streaming_service.initialize_video_writer_early(filename, frame_width, frame_height, original_fps)
        
        # Trigger frame reader with ROI configuration
        if not streaming_service.trigger_frame_reader(file_path, roi_config):
            streaming_service.update_processing_status("error")
            raise HTTPException(status_code=500, detail="Failed to start processing")
        
        return JSONResponse({
            "success": True,
            "message": "Video processing started",
            "filename": filename,
            "status": "processing"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        streaming_service.update_processing_status("error")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Enhanced stats endpoint with frame tracking information"""
    with streaming_service.stats_lock:
        stats_copy = streaming_service.stats.copy()
    
    # Add enhanced tracking information
    stats_copy.update({
        "frame_tracking": {
            "expected_total_frames": streaming_service.expected_total_frames,
            "received_frame_count": streaming_service.received_frame_count,
            "completion_percentage": (streaming_service.received_frame_count / streaming_service.expected_total_frames * 100) if streaming_service.expected_total_frames > 0 else 0,
            "video_finalized": streaming_service.video_finalized,
            "finalization_timer_active": streaming_service.finalization_timer is not None
        }
    })
    
    return JSONResponse(stats_copy)

@app.get("/api/violations")
async def get_violations():
    """Get violation history"""
    """
    get_violations
        Benefit: Returns violation history data for analysis and reporting
        Input: None
        Output: JSONResponse with violation list
        Purpose: Enables frontend to display violation history and analytics
    """
    # This would typically come from your detection service or database
    # For now, return empty list
    return JSONResponse({"violations": []})

@app.get("/api/video-status/{filename}")
async def check_video_status(filename: str):
    """Check if processed video is ready for download"""
    """
    check_video_status
        Benefit: Verifies if processed video is ready for download
        Input: filename (string)
        Output: JSONResponse with file existence and size information
        Purpose: Allows frontend to check processing completion before offering download
    """
    try:
        base_name = os.path.splitext(filename)[0]
        processed_filename = f"processed_{base_name}.mp4"  # Match initialize_video_writer
        file_path = os.path.join(config.PROCESSED_VIDEOS_DIR, processed_filename)
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            return JSONResponse({
                "exists": True,
                "file_size": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "ready": file_size > 0,
                "path": processed_filename
            })
        else:
            return JSONResponse({
                "exists": False,
                "ready": False
            })
            
    except Exception as e:
        return JSONResponse({
            "exists": False,
            "ready": False,
            "error": str(e)
        })

@app.get("/api/download/{filename}")
async def download_processed_video(filename: str):
    """Download processed video file with annotations"""
    """
    download_processed_video
        Benefit: Serves processed video file with proper headers for download
        Input: filename (string)
        Output: FileResponse with video file
        Purpose: Enables users to download annotated video with hygiene monitoring overlays
    """

    try:
        base_name = os.path.splitext(filename)[0]
        processed_filename = f"processed_{base_name}.mp4"  # Match initialize_video_writer
        file_path = os.path.join(config.PROCESSED_VIDEOS_DIR, processed_filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        # Check if file has content
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise HTTPException(status_code=404, detail="Processed video file is empty")
        
        print(f"Downloading: {file_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        # Return file with proper headers
        return FileResponse(
            path=file_path,
            filename=f"processed_{filename}",
            media_type='video/mp4',
            headers={
                "Content-Disposition": f"attachment; filename=processed_{filename}",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with better connection management"""
    await streaming_service.websocket_manager.connect(websocket)
    
    try:
        # Send initial connection status
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to enhanced hygiene monitoring service",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send any buffered frames to new client
        buffer_count = 0
        for frame_data in list(streaming_service.frame_buffer):
            await websocket.send_json(frame_data)
            buffer_count += 1
        
        if buffer_count > 0:
            print(f"Sent {buffer_count} buffered frames to new client")
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Parse client message
                try:
                    client_msg = json.loads(data)
                    msg_type = client_msg.get("type", "unknown")
                    
                    if msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong", 
                            "message": "Connection alive",
                            "timestamp": datetime.now().isoformat()
                        })
                    elif msg_type == "request_stats":
                        # Send current stats directly to this client
                        with streaming_service.stats_lock:
                            stats = streaming_service.stats.copy()
                        await websocket.send_json({
                            "type": "stats_update",
                            "stats": stats,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                except json.JSONDecodeError:
                    # Plain text message - treat as ping
                    await websocket.send_json({
                        "type": "pong", 
                        "message": "Connection alive",
                        "received": data[:50]  # Echo first 50 chars
                    })
                    
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket endpoint error: {e}")
    finally:
        streaming_service.websocket_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """
    health_check
        Benefit: Provides service health status for monitoring and debugging
        Input: None
        Output: JSONResponse with service status
        Purpose: Enables system monitoring and troubleshooting of service availability
    """
    return JSONResponse({"status": "healthy", "service": "hygiene_streaming"})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8003)

