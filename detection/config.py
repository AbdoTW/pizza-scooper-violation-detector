# detection/config.py
import os

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USERNAME = os.getenv('RABBITMQ_USERNAME', 'guest')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'guest')

# Queue Names
RAW_FRAMES_QUEUE = 'raw_frames'
DETECTION_RESULTS_QUEUE = 'detection_results'

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.25
HAND_CONFIDENCE_THRESHOLD = 0.3
SCOOPER_CONFIDENCE_THRESHOLD = 0.3

# Tracking Configuration
HAND_TRACKER_MAX_DISAPPEARED = 90
HAND_TRACKER_MAX_DISTANCE = 200
SCOOPER_TRACKER_MAX_DISAPPEARED = 70
SCOOPER_TRACKER_MAX_DISTANCE = 200

# Violation Detection Configuration
STABILIZATION_PERIOD = 2.0  # seconds
ROI1_TO_ROI2_DELAY = 1.0    # seconds
ROI2_SCOOPER_TRANSITION_DELAY = 1.0  # seconds

# Model Configuration
MODELS_DIR = 'detection/models'
YOLO_MODEL_PATH = "../models/best_scooper_pizzaStoreLarge_largeModel.pt"
FALLBACK_MODEL_PATH = 'yolov8n.pt'

# ROI Configurations for different videos
ROI_CONFIGURATIONS = {
    'video_1': {
        'label': 'Video 1 Configuration',
        'roi1': [[480, 273], [384, 688], [482, 708], [548, 284]],
        'roi2': [[484, 710], [550, 286], [653, 294], [615, 733]]
    },
    'video_2': {
        'label': 'Video 2 Configuration', 
        'roi1': [[402, 264], [289, 678], [396, 699], [478, 271]],
        'roi2': [[398, 701], [480, 273], [605, 277], [561, 718]]
    },
    'video_3': {
        'label': 'Video 3 Configuration',
        'roi1': [[468, 267], [378, 678], [469, 703], [546, 274]],
        'roi2': [[470, 704], [546, 274], [639, 282], [610, 723]]
    },
    'default': {
        'label': 'Default Configuration',
        'roi1': [[468, 267], [378, 678], [469, 703], [546, 274]],
        'roi2': [[470, 704], [546, 274], [639, 282], [610, 723]]
    }
}


# Color Palettes for Visualization
ROI1_HAND_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
ROI1_SCOOPER_COLORS = [(0, 165, 255), (0, 255, 255), (255, 100, 100), (100, 255, 100), (100, 100, 255)]
ROI2_HAND_COLORS = [(128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128)]
ROI2_SCOOPER_COLORS = [(0, 80, 128), (0, 128, 128), (128, 50, 50), (50, 128, 50), (50, 50, 128)]

