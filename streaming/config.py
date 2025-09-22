# Streaming Service Configuration
import os

# Server Configuration
HOST = "localhost"
PORT = 8003

# Directory Configurations
UPLOADS_DIR = "uploads"
TEMP_FRAMES_DIR = "temp_frames" 
PROCESSED_VIDEOS_DIR = "processed_videos"

# RabbitMQ Configuration
RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "guest"
RABBITMQ_PASSWORD = "guest"

# Queue Names (must match detection service)
RAW_FRAMES_QUEUE = "raw_frames"
DETECTION_RESULTS_QUEUE = "detection_results"

# Supported Video Formats
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Video Processing Configuration
OUTPUT_FPS = 20.0
VIDEO_CODEC = 'H264'  # Primary codec
FALLBACK_CODEC = 'XVID'  # Fallback codec

# WebSocket Configuration
MAX_FRAME_BUFFER = 100
BROADCAST_TIMEOUT = 30.0

# Processing Configuration
MAX_FILE_SIZE_MB = 500  # Maximum upload file size
FRAME_PROCESSING_TIMEOUT = 60  # Seconds to wait for frame processing