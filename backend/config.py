"""
VTrackAI Studio Backend Configuration
"""

from pathlib import Path
import os

# Project paths
ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = ROOT_DIR / "uploads"
VTRACK_AI_DIR = Path(__file__).parent.parent.parent / "VTrackAI"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)

# Server settings
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# File upload constraints
MAX_FILE_SIZE_MB = 100
MAX_VIDEO_DURATION = 10  # seconds
MAX_RESOLUTION = (854, 480)  # 480p
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Processing settings
CLEANUP_AFTER_HOURS = 24  # Auto-delete old files after 24 hours
ENABLE_PROGRESS_TRACKING = True

# Model settings (from VTrackAI)
USE_GPU = True
ENABLE_HALF_PRECISION = True

def get_upload_path(filename: str) -> Path:
    """Get full path for uploaded file."""
    return UPLOAD_DIR / filename

def get_output_path(task_id: str, filename: str) -> Path:
    """Get full path for output file."""
    output_dir = UPLOAD_DIR / task_id
    output_dir.mkdir(exist_ok=True)
    return output_dir / filename
