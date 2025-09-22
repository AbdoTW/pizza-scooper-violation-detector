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

import sys
sys.path.append('..')
from shared.rabbitmq_client import RabbitMQClient
from streaming.config import *

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
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
        
        # Ensure directories exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
        os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
    
    def setup_rabbitmq(self):
        """Initialize RabbitMQ connection"""
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
    
    def handle_detection_result(self, ch, method, properties, body):
        """Handle incoming hygiene detection results from RabbitMQ"""
        try:
            data = json.loads(body.decode('utf-8'))
            
            # Check if this indicates end of video processing
            if data.get("end_of_video", False) or data.get("processing_complete", False):
                self.finalize_processed_video()
                print("Video processing completed - saved processed video")
                
                # Send completion signal to frontend
                completion_message = {
                    "type": "processing_complete",
                    "message": "Video processing completed successfully",
                    "timestamp": datetime.now().isoformat(),
                    "total_frames": data.get("total_processed_frames", self.stats["total_frames"]),
                    "video_ready": True
                }
                
                # Broadcast completion to all connected clients
                try:
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast_queue.put(completion_message), 
                            self.loop
                        )
                except Exception as e:
                    print(f"Failed to broadcast completion message: {e}")
                
                # Update status to completed
                with self.stats_lock:
                    self.stats["processing_status"] = "completed"
                
                return  # Don't process this as a regular frame
            
            # Process regular hygiene detection frame
            frame_number = data.get("frame_number", 0)
            hygiene_data = data.get("hygiene_data", {})
            
            # Update hygiene monitoring stats
            with self.stats_lock:
                self.stats["total_frames"] += 1
                self.stats["violation_detected"] = hygiene_data.get("violation_detected", False)
                self.stats["roi1_hand_count"] = hygiene_data.get("roi1_hand_count", 0)
                self.stats["roi2_hand_count"] = hygiene_data.get("roi2_hand_count", 0)
                self.stats["roi1_scooper_count"] = hygiene_data.get("roi1_scooper_count", 0)
                self.stats["roi2_scooper_count"] = hygiene_data.get("roi2_scooper_count", 0)
                self.stats["current_state"] = hygiene_data.get("current_state", "No hands detected")
                self.stats["violation_reason"] = hygiene_data.get("violation_reason", "No violation detected")
                self.stats["stabilization_remaining"] = hygiene_data.get("stabilization_remaining", 0)
                
                # FIX: Update total violations count from detection service
                detection_total_violations = hygiene_data.get("total_violations", 0)
                if detection_total_violations > self.stats["total_violations"]:
                    self.stats["total_violations"] = detection_total_violations
                    print(f"VIOLATION COUNT UPDATE: Backend reports {detection_total_violations} total violations")
                
                # Also check for new violations in the current frame
                if hygiene_data.get("new_violation"):
                    print(f"NEW VIOLATION DETECTED IN FRAME: {hygiene_data.get('new_violation')}")
                    print(f"Current violation status: {hygiene_data.get('violation_detected', False)}")
                    print(f"Streaming service total count: {self.stats['total_violations']}")
            
            # Prepare WebSocket message in the exact format frontend expects
            websocket_message = {
                "type": "frame_update",
                "frame_data": data.get("processed_frame", ""),
                "hygiene_stats": {
                    "violation_detected": hygiene_data.get("violation_detected", False),
                    "roi1_hand_count": hygiene_data.get("roi1_hand_count", 0),
                    "roi2_hand_count": hygiene_data.get("roi2_hand_count", 0),
                    "roi1_scooper_count": hygiene_data.get("roi1_scooper_count", 0),
                    "roi2_scooper_count": hygiene_data.get("roi2_scooper_count", 0),
                    "current_state": hygiene_data.get("current_state", "No hands detected"),
                    "violation_reason": hygiene_data.get("violation_reason", "No violation detected"),
                    "stabilization_remaining": hygiene_data.get("stabilization_remaining", 0),
                    "new_violation": hygiene_data.get("new_violation"),  # Can be None or violation object
                    # FIX: Include the updated total violations count in WebSocket message
                    "total_violations": self.stats["total_violations"]
                }
            }
            
            # Save processed frame for video creation
            self.save_processed_frame(data.get("processed_frame", ""))
            
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
    
    def save_processed_frame(self, base64_frame: str):
        """Save processed frame for video compilation"""
        try:
            if not base64_frame:
                return
            
            # Decode base64 frame
            frame_data = base64.b64decode(base64_frame)
            frame_array = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Warning: Failed to decode frame")
                return
            
            # Initialize video writer on first frame
            if self.video_writer is None:
                self.initialize_video_writer(frame)
            
            # Write frame to video
            if self.video_writer is not None and self.video_writer.isOpened():
                self.video_writer.write(frame)
                
        except Exception as e:
            print(f"Error saving processed frame: {e}")
    
    def initialize_video_writer(self, first_frame):
        """Initialize video writer for processed video"""
        try:
            if self.stats["current_video"]:
                # Create output filename
                base_name = os.path.splitext(self.stats["current_video"])[0]
                self.output_video_path = os.path.join(
                    PROCESSED_VIDEOS_DIR, 
                    f"{base_name}_processed.mp4"
                )
                
                # Get frame dimensions
                height, width = first_frame.shape[:2]
                
                # Use H.264 codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    20.0,  # 20 FPS
                    (width, height)
                )
                
                # Verify video writer is opened
                if not self.video_writer.isOpened():
                    print("H264 codec failed, trying XVID...")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        self.output_video_path,
                        fourcc,
                        20.0,
                        (width, height)
                    )
                
                if self.video_writer.isOpened():
                    print(f"Video writer initialized: {self.output_video_path}")
                    print(f"Dimensions: {width}x{height}")
                else:
                    print(f"Failed to initialize video writer")
                    self.video_writer = None
                    
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            self.video_writer = None
    
    def finalize_processed_video(self):
        """Finalize and save the processed video"""
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
        if self.is_consuming:
            return
        
        if not self.setup_rabbitmq():
            raise Exception("Failed to setup RabbitMQ connection")
        
        self.is_consuming = True
        self.consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.consumer_thread.start()
    
    def _consume_loop(self):
        """RabbitMQ consumption loop"""
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
        """Stop RabbitMQ consumption and finalize video"""
        if self.rabbitmq_client:
            try:
                self.rabbitmq_client.channel.stop_consuming()
                self.rabbitmq_client.close()
            except:
                pass
        self.is_consuming = False
        
        # Finalize processed video when stopping
        if self.video_writer is not None:
            self.finalize_processed_video()
    
    def update_processing_status(self, status: str, video_name: str = None):
        """Update processing status"""
        with self.stats_lock:
            self.stats["processing_status"] = status
            if video_name:
                self.stats["current_video"] = video_name
            if status == "processing" and not self.stats["start_time"]:
                self.stats["start_time"] = datetime.now().isoformat()
    
    def reset_stats(self):
        """Reset processing statistics and video processing"""
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
        
        # Clean up video processing
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except:
                pass
            self.video_writer = None
        
        self.output_video_path = None
        print("Stats and video processing state reset")
    
    def trigger_frame_reader(self, video_path: str, roi_config: dict = None):
        """Trigger frame reader service with ROI configuration"""
        try:
            # Store ROI config for this processing session
            self.current_roi_config = roi_config
            
            # Create command to run frame reader
            frame_reader_path = os.path.join("..", "frame_reader", "frame_reader_service.py")
            
            # Prepare command arguments
            cmd = ["python", frame_reader_path, video_path]
            
            # If ROI config provided, pass it as JSON string
            if roi_config:
                cmd.extend(["--roi-config", json.dumps(roi_config)])
            
            # Run frame reader as subprocess
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"Started frame reader for: {video_path}")
            if roi_config:
                print(f"ROI Config: {roi_config.get('label', 'Custom')}")
            
            return True
        except Exception as e:
            print(f"Failed to trigger frame reader: {e}")
            return False
    
    
    
    async def broadcast_worker(self):
        """Background task to handle WebSocket broadcasting"""
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

app = FastAPI(title="Hygiene Monitoring Streaming Service", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service instance
streaming_service = HygieneStreamingService()

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    try:
        # Validate file format
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Supported: {supported_formats}"
            )
        
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
        
        # Update status
        streaming_service.update_processing_status("processing", filename)
        
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
    """Get current hygiene monitoring statistics"""
    with streaming_service.stats_lock:
        stats_copy = streaming_service.stats.copy()
    
    return JSONResponse(stats_copy)

@app.get("/api/violations")
async def get_violations():
    """Get violation history"""
    # This would typically come from your detection service or database
    # For now, return empty list
    return JSONResponse({"violations": []})

@app.get("/api/video-status/{filename}")
async def check_video_status(filename: str):
    """Check if processed video is ready for download"""
    try:
        base_name = os.path.splitext(filename)[0]
        # Remove UUID prefix if present
        if '_' in base_name:
            base_name = '_'.join(base_name.split('_')[1:])
        
        processed_filename = f"{base_name}_processed.mp4"
        file_path = os.path.join(PROCESSED_VIDEOS_DIR, processed_filename)
        
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
    try:
        base_name = os.path.splitext(filename)[0]
        # Remove UUID prefix if present
        if '_' in base_name:
            base_name = '_'.join(base_name.split('_')[1:])
        
        processed_filename = f"{base_name}_processed.mp4"
        file_path = os.path.join(PROCESSED_VIDEOS_DIR, processed_filename)
        
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
    """WebSocket endpoint for real-time frame streaming"""
    await streaming_service.websocket_manager.connect(websocket)
    
    try:
        # Send any buffered frames to new client
        for frame_data in list(streaming_service.frame_buffer):
            await websocket.send_json(frame_data)
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                # Echo back for connection health check
                await websocket.send_json({"type": "pong", "message": "Connection alive"})
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        streaming_service.websocket_manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "hygiene_streaming"})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8003)