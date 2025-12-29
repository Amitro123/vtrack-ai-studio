"""
SAM3 Video Engine Integration
Unified API for video segmentation and tracking using Meta's SAM3.
Supports both point and text prompts for open-vocabulary segmentation.
GPU-first architecture optimized for Colab deployment.
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

logger = logging.getLogger(__name__)

# Add SAM3 to path
SAM3_PATH = Path(__file__).parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

# Import SAM3 - GPU-first, no fallback
SAM3_AVAILABLE = False
Sam3VideoPredictor = None
try:
    # Import from model_builder (not exported from top-level sam3/__init__.py)
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    SAM3_AVAILABLE = True
    logger.info("SAM3 loaded successfully")
except Exception as e:
    logger.warning(f"SAM3 not available: {e}")


class SAM3VideoEngine:
    """
    GPU-first SAM3 Video Engine.
    Provides video segmentation and tracking using SAM3.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        config: str = "sam3_hiera_l.yaml"
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.predictor = None
        
        logger.info(f"SAM3VideoEngine initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load SAM3 model."""
        if self.predictor is None:
            logger.info(f"Loading SAM3 model from {self.checkpoint_path}...")
            
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found: {self.checkpoint_path}\n"
                    f"Please download it first."
                )
            
            # Use Sam3VideoPredictor class directly with correct API
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=self.checkpoint_path
            )
            
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
        point_px = (
            int(point[0] * width / 100),
            int(point[1] * height / 100)
        )
        
        # Initialize SAM3 session
        session_result = self.predictor.start_session(resource_path=video_path)
        session_id = session_result["session_id"]
        
        try:
            # Add point prompt on the specified frame using session-based API
            self.predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                points=[[float(point_px[0]), float(point_px[1])]],
                point_labels=[1],  # 1 = foreground
            )
            
            # Propagate through video using session-based API
            masks = {}
            for result in self.predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="both",
                start_frame_idx=frame_idx,
                max_frame_num_to_track=-1,  # -1 = all frames
            ):
                frame_index = result["frame_index"]
                outputs = result["outputs"]
                # Extract mask from outputs
                if outputs and len(outputs) > 0:
                    mask_logits = outputs[0].get("mask", None)
                    if mask_logits is not None:
                        mask = (mask_logits > 0).cpu().numpy().astype(np.uint8) * 255
                        masks[frame_index] = mask
        finally:
            # Always close session to free resources
            self.predictor.close_session(session_id)
        
        # Create masked video
        output_path = str(Path(video_path).parent / f"masked_{Path(video_path).stem}.mp4")
        self._create_masked_video(video_path, masks, output_path)
        
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
        Track object from text prompt in video using SAM3's text understanding.
        
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
        
        # Initialize SAM3 session
        session_result = self.predictor.start_session(resource_path=video_path)
        session_id = session_result["session_id"]
        
        try:
            # Add text prompt on the specified frame using session-based API
            self.predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                text=prompt,
            )
            
            # Propagate through video using session-based API
            masks = {}
            for result in self.predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="both",
                start_frame_idx=frame_idx,
                max_frame_num_to_track=-1,  # -1 = all frames
            ):
                frame_index = result["frame_index"]
                outputs = result["outputs"]
                # Extract mask from outputs
                if outputs and len(outputs) > 0:
                    mask_logits = outputs[0].get("mask", None)
                    if mask_logits is not None:
                        mask = (mask_logits > 0).cpu().numpy().astype(np.uint8) * 255
                        masks[frame_index] = mask
        finally:
            # Always close session to free resources
            self.predictor.close_session(session_id)
        
        # Create highlighted video
        output_path = str(Path(video_path).parent / f"highlighted_{Path(video_path).stem}.mp4")
        self._create_highlighted_video(video_path, masks, output_path)
        
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
    
    def remove_object(
        self,
        video_path: str,
        point: Tuple[float, float],
        frame_idx: int = 0
    ) -> Dict:
        """
        Remove object at point from video.
        
        Args:
            video_path: Path to video file
            point: (x, y) in percentage of video dimensions (0-100)
            frame_idx: Frame index to start from
            
        Returns:
            {
                "output_video_path": str,
                "masks": Dict[int, np.ndarray],
                "metadata": Dict
            }
        """
        # First, track the object
        result = self.track_from_point(video_path, point, frame_idx)
        
        # Then apply inpainting to remove it
        output_path = str(Path(video_path).parent / f"removed_{Path(video_path).stem}.mp4")
        self._inpaint_video(video_path, result["masks"], output_path)
        
        return {
            "output_video_path": output_path,
            "masks": result["masks"],
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
        Used by servers that do their own inpainting.
        
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
        output_path: str
    ):
        """Create video with mask overlay."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
                mask_resized = cv2.resize(mask, (width, height))
                
                # Apply green tint to masked region
                overlay = frame.copy()
                overlay[mask_resized > 127] = overlay[mask_resized > 127] * 0.5 + np.array([0, 255, 0]) * 0.5
                frame = overlay.astype(np.uint8)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        logger.info(f"Created masked video: {output_path}")
    
    def _create_highlighted_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str
    ):
        """Create video with object highlighted."""
        self._create_masked_video(video_path, masks, output_path)
    
    def _inpaint_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str
    ):
        """Inpaint video to remove masked objects."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
                mask_resized = cv2.resize(mask, (width, height))
                
                # Simple inpainting
                frame = cv2.inpaint(frame, mask_resized, 3, cv2.INPAINT_TELEA)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        logger.info(f"Created inpainted video: {output_path}")


# Singleton instance
_sam3_engine = None

def get_sam3_engine(checkpoint_path: str, device: str = None) -> SAM3VideoEngine:
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
