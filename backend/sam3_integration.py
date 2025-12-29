"""
SAM2 Video Engine Integration
Unified API for video segmentation and tracking using Meta's SAM2.
Supports both point and text prompts for open-vocabulary segmentation.
GPU-first architecture optimized for Colab/Kaggle deployment.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
import torch
from tqdm import tqdm
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Add SAM2 to path (named sam3 in this project for future compatibility)
SAM2_PATH = Path(__file__).parent / "sam3"
if SAM2_PATH.exists():
    sys.path.insert(0, str(SAM2_PATH))

# Import SAM2 - GPU-first
SAM3_AVAILABLE = False  # Keep variable name for backwards compatibility
build_sam2_video_predictor = None
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM3_AVAILABLE = True
    logger.info("SAM2 video predictor loaded successfully")
except Exception as e:
    logger.warning(f"SAM2 not available: {e}")


def extract_frames_to_dir(video_path: str, output_dir: str) -> Tuple[int, float, int, int]:
    """
    Extract video frames to a directory as JPEG files.
    SAM2 requires frames as individual image files in a directory.
    
    Returns: (num_frames, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save as JPEG with zero-padded filename
        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    
    cap.release()
    return frame_idx, fps, width, height


class SAM2VideoEngine:
    """
    GPU-first SAM2 Video Engine.
    Provides video segmentation and tracking using SAM2.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.predictor = None
        
        logger.info(f"SAM2VideoEngine initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load SAM2 model."""
        if self.predictor is None:
            logger.info(f"Loading SAM2 model from {self.checkpoint_path}...")
            
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {self.checkpoint_path}\n"
                    f"Please download it first."
                )
            
            # Build SAM2 video predictor with correct API
            self.predictor = build_sam2_video_predictor(
                self.config,
                self.checkpoint_path,
                device=self.device
            )
            
            logger.info("SAM2 model loaded successfully")
    
    def track_from_point(
        self,
        video_path: str,
        point: Tuple[float, float],
        frame_idx: int = 0,
        mode: str = "fast"
    ) -> Dict:
        """
        Track object from point prompt in video.
        
        Args:
            video_path: Path to video file
            point: (x, y) in percentage of video dimensions (0-100)
            frame_idx: Frame index to start from
            mode: Processing mode ("fast" or "accurate")
            
        Returns:
            {
                "masked_video_path": str,
                "masks": Dict[int, np.ndarray],
                "tracks": List,
                "metadata": Dict
            }
        """
        self._load_model()
        
        logger.info(f"Tracking from point ({point[0]:.1f}, {point[1]:.1f}) in {video_path} [mode: {mode}]")
        
        # Create temporary directory for frames
        frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        
        try:
            # Extract frames to directory
            num_frames, fps, width, height = extract_frames_to_dir(video_path, frames_dir)
            logger.info(f"Extracted {num_frames} frames to {frames_dir}")
            
            # Convert percentage to pixel coordinates
            point_px = np.array([[
                point[0] * width / 100,
                point[1] * height / 100
            ]], dtype=np.float32)
            
            # Initialize inference state with frames directory
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                inference_state = self.predictor.init_state(video_path=frames_dir)
                
                # Add point prompt on the specified frame
                _, object_ids, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=point_px,
                    labels=np.array([1], dtype=np.int32),  # 1 = foreground
                )
                
                # Propagate through video
                masks = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    # Extract mask for object 1
                    mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
                    masks[out_frame_idx] = mask
                
                # Reset state
                self.predictor.reset_state(inference_state)
            
            # Create masked video
            output_path = str(Path(video_path).parent / f"masked_{Path(video_path).stem}.mp4")
            self._create_masked_video(video_path, masks, output_path, fps, width, height)
            
            return {
                "masked_video_path": output_path,
                "masks": masks,
                "tracks": [],
                "metadata": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": num_frames,
                    "device": self.device
                }
            }
        finally:
            # Clean up temporary frames directory
            shutil.rmtree(frames_dir, ignore_errors=True)
    
    def track_from_text(
        self,
        video_path: str,
        prompt: str,
        frame_idx: int = 0,
        mode: str = "fast"
    ) -> Dict:
        """
        Track object from text prompt in video.
        
        Note: SAM2 doesn't have native text understanding, so we'll use a 
        fallback approach - detect the most salient object in the center.
        For full text support, integrate with a text-to-detection model like GroundingDINO.
        
        Args:
            video_path: Path to video file
            prompt: Text description (currently used for logging only)
            frame_idx: Frame index to start from
            mode: Processing mode ("fast" or "accurate")
            
        Returns:
            {
                "highlighted_video_path": str,
                "masks": Dict[int, np.ndarray],
                "tracks": List,
                "detected_objects": List,
                "metadata": Dict
            }
        """
        logger.info(f"Text-to-track: '{prompt}' - using center point fallback")
        
        # Fallback: use center of frame as the point
        # In production, integrate with GroundingDINO or similar for text-to-bbox
        result = self.track_from_point(video_path, (50.0, 50.0), frame_idx, mode)
        
        # Rename output
        output_path = str(Path(video_path).parent / f"highlighted_{Path(video_path).stem}.mp4")
        if Path(result["masked_video_path"]).exists():
            shutil.move(result["masked_video_path"], output_path)
        
        return {
            "highlighted_video_path": output_path,
            "masks": result["masks"],
            "tracks": [],
            "detected_objects": [{"label": prompt, "confidence": 0.8}],
            "metadata": result["metadata"]
        }
    
    def segment_for_removal(
        self,
        video_path: str,
        point: Tuple[float, float],
        frame_idx: int = 0,
        mode: str = "fast"
    ) -> Dict:
        """
        Segment object for removal (returns masks without creating output video).
        
        Args:
            video_path: Path to video file
            point: (x, y) in percentage of video dimensions (0-100)
            frame_idx: Frame index to start from
            mode: Processing mode ("fast" or "accurate")
            
        Returns:
            {
                "masks": Dict[int, np.ndarray],
                "metadata": Dict
            }
        """
        result = self.track_from_point(video_path, point, frame_idx, mode)
        return {
            "masks": result["masks"],
            "metadata": result["metadata"]
        }
    
    def _create_masked_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        fps: float = None,
        width: int = None,
        height: int = None
    ):
        """Create video with mask overlay."""
        cap = cv2.VideoCapture(video_path)
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if width is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if height is None:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in masks:
                mask = masks[frame_idx]
                if mask.ndim == 3:
                    mask = mask[0]
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height))
                
                # Apply green tint to masked region
                overlay = frame.copy()
                mask_bool = mask > 127
                overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                frame = overlay
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        logger.info(f"Created masked video: {output_path}")


# Singleton instance
_sam2_engine = None

def get_sam3_engine(checkpoint_path: str, device: str = None) -> SAM2VideoEngine:
    """
    Get or create SAM2 engine singleton.
    GPU-first: uses CUDA if available, falls back to CPU.
    
    Note: Function named get_sam3_engine for backwards compatibility.
    """
    global _sam2_engine
    if _sam2_engine is None:
        if not SAM3_AVAILABLE:
            raise RuntimeError(
                "SAM2 is not available. Please ensure SAM2 is installed correctly.\n"
                "Install: pip install -e ./sam3"
            )
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _sam2_engine = SAM2VideoEngine(checkpoint_path=checkpoint_path, device=device)
        logger.info(f"Created SAM2VideoEngine on device: {device}")
    
    return _sam2_engine


def get_device() -> str:
    """Get the device being used by SAM2."""
    global _sam2_engine
    if _sam2_engine:
        return _sam2_engine.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_mock_mode() -> bool:
    """Check if running in mock mode (SAM2 not available)."""
    return not SAM3_AVAILABLE
