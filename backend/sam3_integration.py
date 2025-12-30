"""
SAM3 Video Engine Integration
Unified API for video segmentation and tracking using Meta's SAM3.
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
import shutil

logger = logging.getLogger(__name__)

# Add SAM3 to path
SAM3_PATH = Path(__file__).parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

# Import SAM3 - GPU-first
SAM3_AVAILABLE = False
build_sam3_video_predictor = None
try:
    from sam3.model_builder import build_sam3_video_predictor
    SAM3_AVAILABLE = True
    logger.info("SAM3 video predictor loaded successfully")
except Exception as e:
    logger.warning(f"SAM3 not available: {e}")


class SAM3VideoEngine:
    """
    GPU-first SAM3 Video Engine.
    Provides video segmentation and tracking using SAM3.
    Uses the handle_request API pattern.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        config: str = None
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.predictor = None
        
        logger.info(f"SAM3VideoEngine initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load SAM3 model."""
        if self.predictor is None:
            logger.info("Loading SAM3 video predictor...")
            
            # Build SAM3 video predictor
            self.predictor = build_sam3_video_predictor()
            
            logger.info("SAM3 model loaded successfully")
    
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
        
        # Read video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Convert percentage to pixel coordinates
        point_px = [
            point[0] * width / 100,
            point[1] * height / 100
        ]
        
        # Start a session with SAM3
        with torch.inference_mode():
            # Start session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            session_id = response["session_id"]
            
            try:
                # Add point prompt
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        obj_id=1,
                        points=[point_px],
                        point_labels=[1],  # 1 = foreground
                    )
                )
                
                # Get masks from outputs
                masks = {}
                if "outputs" in response and response["outputs"]:
                    for output in response["outputs"]:
                        frame_index = output.get("frame_index", frame_idx)
                        mask_data = output.get("mask", output.get("masks", None))
                        if mask_data is not None:
                            if torch.is_tensor(mask_data):
                                mask = (mask_data > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
                            else:
                                mask = (np.array(mask_data) > 0).astype(np.uint8) * 255
                            masks[frame_index] = mask
                
                # Propagate through video
                response = self.predictor.handle_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_idx=frame_idx,
                    )
                )
                
                # Collect propagated masks
                if "outputs" in response:
                    for output in response["outputs"]:
                        frame_index = output.get("frame_index")
                        mask_data = output.get("mask", output.get("masks", None))
                        if frame_index is not None and mask_data is not None:
                            if torch.is_tensor(mask_data):
                                mask = (mask_data > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
                            else:
                                mask = (np.array(mask_data) > 0).astype(np.uint8) * 255
                            masks[frame_index] = mask
                
            finally:
                # Close session
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
        
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
                "total_frames": total_frames,
                "device": self.device
            }
        }
    
    def track_from_text(
        self,
        video_path: str,
        prompt: str,
        frame_idx: int = 0,
        mode: str = "fast"
    ) -> Dict:
        """
        Track object from text prompt in video using SAM3's native text understanding.
        
        Args:
            video_path: Path to video file
            prompt: Text description (e.g., "drummer", "person on the left")
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
        self._load_model()
        
        logger.info(f"Tracking from text prompt: '{prompt}' in {video_path} [mode: {mode}]")
        
        # Read video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Start a session with SAM3
        with torch.inference_mode():
            # Start session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            session_id = response["session_id"]
            
            try:
                # Add text prompt - SAM3's native text understanding
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        text=prompt,
                    )
                )
                
                # Get masks from outputs
                masks = {}
                if "outputs" in response and response["outputs"]:
                    for output in response["outputs"]:
                        frame_index = output.get("frame_index", frame_idx)
                        mask_data = output.get("mask", output.get("masks", None))
                        if mask_data is not None:
                            if torch.is_tensor(mask_data):
                                mask = (mask_data > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
                            else:
                                mask = (np.array(mask_data) > 0).astype(np.uint8) * 255
                            masks[frame_index] = mask
                
                # Propagate through video
                response = self.predictor.handle_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_idx=frame_idx,
                    )
                )
                
                # Collect propagated masks
                if "outputs" in response:
                    for output in response["outputs"]:
                        frame_index = output.get("frame_index")
                        mask_data = output.get("mask", output.get("masks", None))
                        if frame_index is not None and mask_data is not None:
                            if torch.is_tensor(mask_data):
                                mask = (mask_data > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
                            else:
                                mask = (np.array(mask_data) > 0).astype(np.uint8) * 255
                            masks[frame_index] = mask
                
            finally:
                # Close session
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
        
        # Create highlighted video
        output_path = str(Path(video_path).parent / f"highlighted_{Path(video_path).stem}.mp4")
        self._create_masked_video(video_path, masks, output_path, fps, width, height)
        
        return {
            "highlighted_video_path": output_path,
            "masks": masks,
            "tracks": [],
            "detected_objects": [{"label": prompt, "confidence": 1.0}],
            "metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "device": self.device
            }
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
_sam3_engine = None

def get_sam3_engine(checkpoint_path: str = None, device: str = None) -> SAM3VideoEngine:
    """
    Get or create SAM3 engine singleton.
    GPU-first: uses CUDA if available, falls back to CPU.
    """
    global _sam3_engine
    if _sam3_engine is None:
        if not SAM3_AVAILABLE:
            raise RuntimeError(
                "SAM3 is not available. Please ensure SAM3 is installed correctly.\n"
                "Install: pip install -e ./sam3"
            )
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _sam3_engine = SAM3VideoEngine(checkpoint_path=checkpoint_path, device=device)
        logger.info(f"Created SAM3VideoEngine on device: {device}")
    
    return _sam3_engine


def get_device() -> str:
    """Get the device being used by SAM3."""
    global _sam3_engine
    if _sam3_engine:
        return _sam3_engine.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_mock_mode() -> bool:
    """Check if running in mock mode (SAM3 not available)."""
    return not SAM3_AVAILABLE
