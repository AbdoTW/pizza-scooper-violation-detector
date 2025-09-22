# Frame Reader Configuration

# RabbitMQ Settings
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = 'guest'
RABBITMQ_PASSWORD = 'guest'

# Queue Names
RAW_FRAMES_QUEUE = 'raw_frames'
DETECTION_RESULTS_QUEUE = 'detection_results'

# Video Processing Settings
FRAME_RATE = 10  # Extract 10 frames per second
MAX_FRAME_WIDTH = 1280  # Resize frames for faster processing
MAX_FRAME_HEIGHT = 720

# File Paths
UPLOADS_DIR = 'uploads'
TEMP_FRAMES_DIR = 'temp_frames'

# Supported video formats
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']