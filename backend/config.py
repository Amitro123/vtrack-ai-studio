"""
VTrackAI Studio Backend Configuration (SAM3 Version)
"""

from pathlib import Path
import os
import torch

# Project paths
ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = ROOT_DIR / "uploads"
SAM3_DIR = ROOT_DIR / "sam3"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Server settings
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", 8000))

# CORS settings
# For cloud deployments (Colab/Kaggle with tunnels), allow all origins
CORS_ALLOW_ALL = os.getenv("CORS_ALLOW_ALL", "false").lower() in ("true", "1", "yes")

CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server (default)
    "http://localhost:8080",  # Vite dev server (actual port)
    "http://localhost:4173",  # Vite preview
    "http://localhost:3000",  # Alternative
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:3000",
]

# Add tunnel domains if in cloud environment
if CORS_ALLOW_ALL or "COLAB_GPU" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    CORS_ORIGINS = ["*"]  # Allow all origins for tunnel access

# File upload constraints
MAX_FILE_SIZE_MB = 100
MAX_VIDEO_DURATION = 10  # seconds
MAX_RESOLUTION = (1280, 720)  # 720p
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Processing settings
CLEANUP_AFTER_HOURS = 24  # Auto-delete old files after 24 hours
ENABLE_PROGRESS_TRACKING = True

# Processing modes (for SAM3 optimization)
PROCESSING_FPS_FAST = 5.0       # Fast mode: 5 FPS processing
PROCESSING_FPS_ACCURATE = 10.0  # Accurate mode: 10 FPS processing
KEYFRAMES_FAST = 12             # Fast mode: 12 keyframes
KEYFRAMES_ACCURATE = 32         # Accurate mode: 32 keyframes

# SAM3 settings
SAM3_CHECKPOINT = CHECKPOINTS_DIR / "sam3" / "sam3_hiera_large.pt"
SAM3_CONFIG = "sam3_hiera_l.yaml"

# Audio settings (Demucs - unchanged)
DEMUCS_MODEL = "htdemucs"
AUDIO_SAMPLE_RATE = 44100

# GPU settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = True if DEVICE == "cuda" else False
MAX_VRAM_GB = 16  # SAM3 requires more VRAM than SAM2

# Colab-specific settings
if "COLAB_GPU" in os.environ:
    DEVICE = "cuda"
    MAX_VRAM_GB = 16  # Colab T4/A100

def get_upload_path(filename: str) -> Path:
    """Get full path for uploaded file."""
    return UPLOAD_DIR / filename

def get_output_path(task_id: str, filename: str) -> Path:
    """Get full path for output file."""
    output_dir = UPLOAD_DIR / task_id
    output_dir.mkdir(exist_ok=True)
    return output_dir / filename

def validate_sam3_setup() -> tuple[bool, str]:
    """
    Validate SAM3 installation and checkpoint.
    
    Returns:
        (is_valid, message)
    """
    # Check if SAM3 directory exists
    if not SAM3_DIR.exists():
        return False, (
            f"SAM3 not found at {SAM3_DIR}\n"
            f"Please install: cd backend && git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e ."
        )
    
    # Check if checkpoint exists
    if not SAM3_CHECKPOINT.exists():
        return False, (
            f"SAM3 checkpoint not found: {SAM3_CHECKPOINT}\n"
            f"Please download: mkdir -p {SAM3_CHECKPOINT.parent} && "
            f"wget -O {SAM3_CHECKPOINT} https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt"
        )
    
    # Check CUDA availability
    if DEVICE == "cpu":
        return True, "[WARNING] Running on CPU (slow). GPU recommended for SAM3."
    
    return True, f"[SUCCESS] SAM3 setup valid (device: {DEVICE}, VRAM: {MAX_VRAM_GB}GB)"
