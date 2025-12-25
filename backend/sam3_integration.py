"""
SAM3 Video Engine Integration
Unified API for video segmentation and tracking using Meta's SAM3.
Supports both point and text prompts for open-vocabulary segmentation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import cv2
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Add SAM3 to path
SAM3_PATH = Path(__file__).parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

try:
    from sam3 import build_sam3_video_predictor
    from sam3.utils.video import VideoReader
    SAM3_AVAILABLE = True
    logger.info("SAM3 loaded successfully")
except ImportError as e:
    logger.warning(f"SAM3 not available: {e}")
    logger.info("Please install SAM3: cd backend && git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .")
    SAM3_AVAILABLE = False


class SAM3VideoEngine:
    """
    Unified SAM3 engine for video segmentation and tracking.
    Supports both point and text prompts with native video understanding.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: str = "sam3_hiera_l.yaml",
        device: str = "cuda"
    ):
        """
        Initialize SAM3 video predictor.
        
        Args:
            checkpoint_path: Path to SAM3 checkpoint (.pt file)
            config: SAM3 model config (sam3_hiera_l.yaml, sam3_hiera_b+.yaml, etc.)
            device: Device to run on ('cuda' or 'cpu')
        """
        if not SAM3_AVAILABLE:
            raise RuntimeError("SAM3 is not installed. Please install it first.")
        
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        self.predictor = None
        
        # Lazy loading - only load when needed
        logger.info(f"SAM3VideoEngine initialized (device: {device})")
    
    def _load_model(self):
        """Lazy load SAM3 model."""
        if self.predictor is None:
            logger.info(f"Loading SAM3 model from {self.checkpoint_path}...")
            
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found: {self.checkpoint_path}\n"
                    f"Please download it first."
                )
            
            self.predictor = build_sam3_video_predictor(
                config_file=self.config,
                ckpt_path=self.checkpoint_path,
                device=self.device
            )
            
            logger.info("SAM3 model loaded successfully")
    
    def track_from_point(
        self,
        video_path: str,
        point: Tuple[float, float],
        frame_idx: int = 0
    ) -> Dict:
        """
        Track object from point prompt in video.
        
        Args:
            video_path: Path to input video
            point: (x, y) coordinates as percentages (0-100)
            frame_idx: Frame index to start tracking from
            
        Returns:
            {
                "masked_video_path": str,
                "masks": Dict[int, np.ndarray],
                "tracks": List,
                "metadata": Dict
            }
        """
        self._load_model()
        
        logger.info(f"Tracking from point ({point[0]:.1f}, {point[1]:.1f}) in {video_path}")
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert percentage to pixel coordinates
        point_px = (
            int(point[0] * width / 100),
            int(point[1] * height / 100)
        )
        
        # Initialize SAM3 inference state
        inference_state = self.predictor.init_state(video_path=video_path)
        
        # Add point prompt on the specified frame
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=np.array([[point_px[0], point_px[1]]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),  # 1 = foreground
        )
        
        # Propagate masks through the video
        logger.info("Propagating masks through video...")
        masks = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state
        ):
            # Get mask for the tracked object
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            masks[out_frame_idx] = mask
        
        # Create masked video with overlay
        output_path = str(Path(video_path).parent / f"masked_{Path(video_path).name}")
        self._create_masked_video(
            video_path,
            masks,
            output_path,
            color=(255, 0, 0),  # Red mask
            alpha=0.5
        )
        
        cap.release()
        
        # Clear GPU cache
        self._clear_cache()
        
        return {
            "masked_video_path": output_path,
            "masks": {int(k): v for k, v in masks.items()},
            "tracks": [{"obj_id": 1, "frames": list(masks.keys())}],
            "metadata": {
                "fps": fps,
                "resolution": (width, height),
                "total_frames": total_frames,
                "tracked_frames": len(masks)
            }
        }
    
    def track_from_text(
        self,
        video_path: str,
        prompt: str,
        frame_idx: int = 0
    ) -> Dict:
        """
        Track object from text prompt in video.
        Uses SAM3's native text understanding (no separate detection model needed).
        
        Args:
            video_path: Path to input video
            prompt: Text description (e.g., "drummer", "person on the left")
            frame_idx: Frame index to start from
            
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
        
        logger.info(f"Tracking from text prompt: '{prompt}' in {video_path}")
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize SAM3 inference state
        inference_state = self.predictor.init_state(video_path=video_path)
        
        # Add text prompt on the specified frame
        # SAM3 natively supports text prompts
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_text_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            text=prompt
        )
        
        # Propagate masks through the video
        logger.info("Propagating masks through video...")
        masks = {}
        detected_objects = []
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state
        ):
            # Get mask for the detected object
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            masks[out_frame_idx] = mask
            
            # Track detected object info
            if out_frame_idx == frame_idx:
                detected_objects.append({
                    "obj_id": 1,
                    "text": prompt,
                    "confidence": float(out_mask_logits[0].max())
                })
        
        # Create highlighted video with overlay
        output_path = str(Path(video_path).parent / f"highlighted_{Path(video_path).name}")
        self._create_masked_video(
            video_path,
            masks,
            output_path,
            color=(0, 255, 0),  # Green highlight
            alpha=0.3
        )
        
        cap.release()
        
        # Clear GPU cache
        self._clear_cache()
        
        return {
            "highlighted_video_path": output_path,
            "masks": {int(k): v for k, v in masks.items()},
            "tracks": [{"obj_id": 1, "frames": list(masks.keys()), "text": prompt}],
            "detected_objects": detected_objects,
            "metadata": {
                "fps": fps,
                "resolution": (width, height),
                "total_frames": total_frames,
                "tracked_frames": len(masks),
                "prompt": prompt
            }
        }
    
    def segment_for_removal(
        self,
        video_path: str,
        point: Tuple[float, float],
        frame_idx: int = 0
    ) -> Dict:
        """
        Segment object for removal/inpainting.
        
        Args:
            video_path: Path to input video
            point: (x, y) coordinates as percentages (0-100)
            frame_idx: Frame index to start from
            
        Returns:
            {
                "masks": Dict[int, np.ndarray],
                "tracks": List
            }
        """
        # Use same logic as track_from_point but only return masks
        result = self.track_from_point(video_path, point, frame_idx)
        
        return {
            "masks": result["masks"],
            "tracks": result["tracks"]
        }
    
    def _create_masked_video(
        self,
        input_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5
    ):
        """
        Create video with mask overlay.
        
        Args:
            input_path: Input video path
            masks: Dictionary of frame_idx -> mask
            output_path: Output video path
            color: RGB color for mask overlay
            alpha: Transparency (0-1)
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        pbar = tqdm(total=len(masks), desc="Creating masked video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply mask overlay if available for this frame
            if frame_idx in masks:
                mask = masks[frame_idx]
                
                # Resize mask if needed
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height))
                
                # Create colored overlay
                overlay = frame.copy()
                overlay[mask > 0] = color
                
                # Blend with original frame
                frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
                
                pbar.update(1)
            
            out.write(frame)
            frame_idx += 1
        
        pbar.close()
        cap.release()
        out.release()
        
        logger.info(f"Masked video saved to {output_path}")
    
    def _clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on deletion."""
        self._clear_cache()


# Singleton instance
_sam3_engine = None

def get_sam3_engine(checkpoint_path: str, device: str = "cuda") -> SAM3VideoEngine:
    """Get or create SAM3 engine singleton."""
    global _sam3_engine
    if _sam3_engine is None:
        _sam3_engine = SAM3VideoEngine(checkpoint_path=checkpoint_path, device=device)
    return _sam3_engine
