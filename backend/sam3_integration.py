"""
SAM3 Video Engine Integration
Unified API for video segmentation and tracking using Meta's SAM3.
Supports the two-step workflow: propagate → track.
GPU-first architecture optimized for Colab/Kaggle deployment.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2
import torch
import shutil

logger = logging.getLogger(__name__)

# Add SAM3 to path
SAM3_PATH = Path(__file__).parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

# Cache directory for propagation results
CACHE_DIR = Path(__file__).parent / "sam3_cache"
CACHE_DIR.mkdir(exist_ok=True)

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
    Supports two-step workflow: propagate → track.
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
        self._active_sessions: Dict[str, Any] = {}  # session_id -> session data
        
        logger.info(f"SAM3VideoEngine initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load SAM3 model."""
        if self.predictor is None:
            logger.info("Loading SAM3 video predictor...")
            self.predictor = build_sam3_video_predictor()
            logger.info("SAM3 model loaded successfully")
    
    def propagate_video(self, video_path: str) -> Dict:
        """
        Step 1: Pre-compute SAM3 features/cache for video.
        Must be called before track_point.
        
        Args:
            video_path: Path to video file
            
        Returns:
            {
                "session_id": str,
                "cache_path": str,
                "num_frames": int,
                "metadata": Dict
            }
        """
        self._load_model()
        
        logger.info(f"Propagating video: {video_path}")
        
        # Read video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Generate session ID
        import hashlib
        session_id = hashlib.md5(f"{video_path}_{os.path.getmtime(video_path)}".encode()).hexdigest()[:16]
        
        # Create cache directory
        cache_path = CACHE_DIR / session_id
        cache_path.mkdir(exist_ok=True)
        
        with torch.inference_mode():
            # Start SAM3 session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            sam3_session_id = response["session_id"]
            
            # Store session info
            self._active_sessions[session_id] = {
                "sam3_session_id": sam3_session_id,
                "video_path": video_path,
                "cache_path": str(cache_path),
                "metadata": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "device": self.device
                }
            }
            
            # Save session info to cache
            import json
            with open(cache_path / "session.json", "w") as f:
                json.dump({
                    "session_id": session_id,
                    "sam3_session_id": sam3_session_id,
                    "video_path": video_path,
                    "metadata": self._active_sessions[session_id]["metadata"]
                }, f)
        
        logger.info(f"Propagation complete. Session ID: {session_id}")
        
        return {
            "session_id": session_id,
            "cache_path": str(cache_path),
            "num_frames": total_frames,
            "metadata": self._active_sessions[session_id]["metadata"]
        }
    
    def track_point(
        self,
        session_id: str,
        point: Tuple[float, float],
        frame_idx: int = 0,
        obj_id: int = 1
    ) -> Dict:
        """
        Step 2: Track point using pre-computed propagation cache.
        Must call propagate_video first.
        
        Args:
            session_id: Session ID from propagate_video
            point: (x, y) in percentage of video dimensions (0-100)
            frame_idx: Frame index for the click
            obj_id: Object ID for tracking
            
        Returns:
            {
                "masks": Dict[int, np.ndarray],
                "masked_video_path": str,
                "metadata": Dict
            }
        """
        self._load_model()
        
        # Load session from cache if not in memory
        if session_id not in self._active_sessions:
            cache_path = CACHE_DIR / session_id
            if not cache_path.exists():
                raise ValueError(f"Session {session_id} not found. Call propagate_video first.")
            
            import json
            with open(cache_path / "session.json") as f:
                session_data = json.load(f)
            
            # Re-start SAM3 session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=session_data["video_path"],
                )
            )
            
            self._active_sessions[session_id] = {
                "sam3_session_id": response["session_id"],
                "video_path": session_data["video_path"],
                "cache_path": str(cache_path),
                "metadata": session_data["metadata"]
            }
        
        session = self._active_sessions[session_id]
        sam3_session_id = session["sam3_session_id"]
        metadata = session["metadata"]
        
        # Convert percentage to pixel coordinates
        point_px = [
            point[0] * metadata["width"] / 100,
            point[1] * metadata["height"] / 100
        ]
        
        logger.info(f"Tracking point ({point[0]:.1f}, {point[1]:.1f}) in session {session_id}")
        
        masks = {}
        
        with torch.inference_mode():
            # Add point prompt
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=sam3_session_id,
                    frame_index=frame_idx,
                    obj_id=obj_id,
                    points=[point_px],
                    point_labels=[1],  # 1 = foreground
                )
            )
            
            # Get initial mask
            if "outputs" in response and response["outputs"]:
                for output in response["outputs"]:
                    frame_index = output.get("frame_index", frame_idx)
                    mask_data = output.get("mask", output.get("masks", None))
                    if mask_data is not None:
                        mask = self._process_mask(mask_data)
                        masks[frame_index] = mask
            
            # Propagate through video
            response = self.predictor.handle_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=sam3_session_id,
                    start_frame_idx=frame_idx,
                )
            )
            
            # Collect propagated masks
            if "outputs" in response:
                for output in response["outputs"]:
                    frame_index = output.get("frame_index")
                    mask_data = output.get("mask", output.get("masks", None))
                    if frame_index is not None and mask_data is not None:
                        mask = self._process_mask(mask_data)
                        masks[frame_index] = mask
        
        # Create masked video
        video_path = session["video_path"]
        output_path = str(Path(video_path).parent / f"masked_{Path(video_path).stem}.mp4")
        self._create_masked_video(
            video_path, masks, output_path,
            metadata["fps"], metadata["width"], metadata["height"]
        )
        
        return {
            "masks": masks,
            "masked_video_path": output_path,
            "metadata": metadata
        }
    
    def track_text(
        self,
        session_id: str,
        prompt: str,
        frame_idx: int = 0,
        obj_id: int = 1
    ) -> Dict:
        """
        Track object from text prompt using pre-computed propagation cache.
        
        Args:
            session_id: Session ID from propagate_video
            prompt: Text description
            frame_idx: Frame index for the prompt
            obj_id: Object ID for tracking
            
        Returns:
            {
                "masks": Dict[int, np.ndarray],
                "highlighted_video_path": str,
                "metadata": Dict
            }
        """
        self._load_model()
        
        # Load session
        if session_id not in self._active_sessions:
            cache_path = CACHE_DIR / session_id
            if not cache_path.exists():
                raise ValueError(f"Session {session_id} not found. Call propagate_video first.")
            
            import json
            with open(cache_path / "session.json") as f:
                session_data = json.load(f)
            
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=session_data["video_path"],
                )
            )
            
            self._active_sessions[session_id] = {
                "sam3_session_id": response["session_id"],
                "video_path": session_data["video_path"],
                "cache_path": str(cache_path),
                "metadata": session_data["metadata"]
            }
        
        session = self._active_sessions[session_id]
        sam3_session_id = session["sam3_session_id"]
        metadata = session["metadata"]
        
        logger.info(f"Tracking text '{prompt}' in session {session_id}")
        
        masks = {}
        
        with torch.inference_mode():
            # Add text prompt
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=sam3_session_id,
                    frame_index=frame_idx,
                    obj_id=obj_id,
                    text=prompt,
                )
            )
            
            # Get initial mask
            if "outputs" in response and response["outputs"]:
                for output in response["outputs"]:
                    frame_index = output.get("frame_index", frame_idx)
                    mask_data = output.get("mask", output.get("masks", None))
                    if mask_data is not None:
                        mask = self._process_mask(mask_data)
                        masks[frame_index] = mask
            
            # Propagate
            response = self.predictor.handle_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=sam3_session_id,
                    start_frame_idx=frame_idx,
                )
            )
            
            if "outputs" in response:
                for output in response["outputs"]:
                    frame_index = output.get("frame_index")
                    mask_data = output.get("mask", output.get("masks", None))
                    if frame_index is not None and mask_data is not None:
                        mask = self._process_mask(mask_data)
                        masks[frame_index] = mask
        
        # Create highlighted video
        video_path = session["video_path"]
        output_path = str(Path(video_path).parent / f"highlighted_{Path(video_path).stem}.mp4")
        self._create_masked_video(
            video_path, masks, output_path,
            metadata["fps"], metadata["width"], metadata["height"]
        )
        
        return {
            "masks": masks,
            "highlighted_video_path": output_path,
            "detected_objects": [{"label": prompt, "confidence": 1.0}],
            "metadata": metadata
        }
    
    def close_session(self, session_id: str):
        """Close a propagation session and free resources."""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            try:
                self.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session["sam3_session_id"],
                    )
                )
            except Exception as e:
                logger.warning(f"Error closing SAM3 session: {e}")
            del self._active_sessions[session_id]
    
    def _process_mask(self, mask_data) -> np.ndarray:
        """Process mask data to numpy array."""
        if torch.is_tensor(mask_data):
            mask = (mask_data > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
        else:
            mask = (np.array(mask_data) > 0).astype(np.uint8) * 255
        return mask
    
    def _create_masked_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        fps: float,
        width: int,
        height: int
    ):
        """Create video with mask overlay."""
        cap = cv2.VideoCapture(video_path)
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
    
    # Legacy methods for backwards compatibility
    def track_from_point(self, video_path: str, point: Tuple[float, float], frame_idx: int = 0, mode: str = "fast") -> Dict:
        """Legacy method - combines propagate + track."""
        result = self.propagate_video(video_path)
        session_id = result["session_id"]
        try:
            track_result = self.track_point(session_id, point, frame_idx)
            return {
                "masked_video_path": track_result["masked_video_path"],
                "masks": track_result["masks"],
                "tracks": [],
                "metadata": track_result["metadata"]
            }
        finally:
            self.close_session(session_id)
    
    def track_from_text(self, video_path: str, prompt: str, frame_idx: int = 0, mode: str = "fast") -> Dict:
        """Legacy method - combines propagate + track."""
        result = self.propagate_video(video_path)
        session_id = result["session_id"]
        try:
            track_result = self.track_text(session_id, prompt, frame_idx)
            return {
                "highlighted_video_path": track_result["highlighted_video_path"],
                "masks": track_result["masks"],
                "tracks": [],
                "detected_objects": track_result["detected_objects"],
                "metadata": track_result["metadata"]
            }
        finally:
            self.close_session(session_id)
    
    def segment_for_removal(self, video_path: str, point: Tuple[float, float], frame_idx: int = 0, mode: str = "fast") -> Dict:
        """Legacy method for removal."""
        result = self.track_from_point(video_path, point, frame_idx, mode)
        return {"masks": result["masks"], "metadata": result["metadata"]}


# Singleton instance
_sam3_engine = None

def get_sam3_engine(checkpoint_path: str = None, device: str = None) -> SAM3VideoEngine:
    """Get or create SAM3 engine singleton."""
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
