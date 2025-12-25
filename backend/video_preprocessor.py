"""
Video preprocessing utilities for SAM3 optimization
Includes FPS downsampling, keyframe selection, and mask interpolation
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
import time

from config import (
    PROCESSING_FPS_FAST,
    PROCESSING_FPS_ACCURATE,
    KEYFRAMES_FAST,
    KEYFRAMES_ACCURATE
)

# Global frame cache: {video_hash: {"frames": [...], "fps": float, "timestamp": float}}
_FRAME_CACHE: Dict[str, Dict] = {}
_CACHE_TTL = 3600  # Cache frames for 1 hour

# Threshold for dense processing (videos shorter than this will use all frames)
DENSE_PROCESSING_THRESHOLD_SECONDS = 3.0
logger = logging.getLogger(__name__)


def get_video_hash(video_path: str) -> str:
    """Generate a hash for video file to use as cache key"""
    path = Path(video_path)
    # Use file path + size + mtime for quick hash
    stat = path.stat()
    hash_input = f"{path.absolute()}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def cleanup_old_cache():
    """Remove cache entries older than TTL"""
    current_time = time.time()
    to_remove = []
    for key, value in _FRAME_CACHE.items():
        if current_time - value.get("timestamp", 0) > _CACHE_TTL:
            to_remove.append(key)
    
    for key in to_remove:
        del _FRAME_CACHE[key]


def load_video_frames(video_path: str, use_cache: bool = True) -> Tuple[List[np.ndarray], float]:
    """
    Load all frames from video into memory.
    Uses caching to avoid re-reading the same video.
    
    Args:
        video_path: Path to video file
        use_cache: Whether to use frame cache
    
    Returns:
        Tuple of (frames list, fps)
    """
    # Check cache first
    if use_cache:
        cleanup_old_cache()
        video_hash = get_video_hash(video_path)
        if video_hash in _FRAME_CACHE:
            cache_entry = _FRAME_CACHE[video_hash]
            return cache_entry["frames"], cache_entry["fps"]
    
    # Load frames from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # Cache the frames
    if use_cache:
        _FRAME_CACHE[video_hash] = {
            "frames": frames,
            "fps": fps,
            "timestamp": time.time()
        }
    
    return frames, fps


def should_use_dense_processing(video_path: str, target_fps: float) -> bool:
    """
    Determine if video is short enough to process all frames densely.
    
    Args:
        video_path: Path to video file
        target_fps: Target FPS for processing
    
    Returns:
        True if video should use dense processing (all frames)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0:
        return False
    
    duration = frame_count / fps
    return duration <= DENSE_PROCESSING_THRESHOLD_SECONDS


def downsample_video(
    video_path: str,
    target_fps: float
) -> Tuple[str, Dict[int, int]]:
    """
    Downsample video to target FPS for efficient processing.
    
    Args:
        video_path: Path to input video
        target_fps: Target FPS (if None, returns original video)
        
    Returns:
        Tuple of (downsampled_video_path, frame_mapping)
        - downsampled_video_path: Path to temporary downsampled video
        - frame_mapping: Dict mapping processed frame idx -> original frame idx
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If no target FPS or target >= original, return original video
    if target_fps is None or target_fps >= original_fps:
        cap.release()
        # Identity mapping: processed frame i -> original frame i
        frame_mapping = {i: i for i in range(total_frames)}
        return video_path, frame_mapping
    
    logger.info(f"Downsampling video from {original_fps:.1f} FPS to {target_fps:.1f} FPS")
    
    # Calculate frame skip interval
    frame_interval = int(original_fps / target_fps)
    
    # Create temporary output file
    temp_dir = Path(video_path).parent
    temp_file = tempfile.NamedTemporaryFile(
        suffix='.mp4',
        dir=temp_dir,
        delete=False
    )
    output_path = temp_file.name
    temp_file.close()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_mapping = {}
    processed_idx = 0
    original_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep every Nth frame
        if original_idx % frame_interval == 0:
            out.write(frame)
            frame_mapping[processed_idx] = original_idx
            processed_idx += 1
        
        original_idx += 1
    
    cap.release()
    out.release()
    
    logger.info(f"Downsampled video: {total_frames} -> {processed_idx} frames")
    logger.info(f"Saved to: {output_path}")
    
    return output_path, frame_mapping


def select_keyframes(num_frames: int, k: int) -> List[int]:
    """
    Select keyframe indices for SAM3 processing using uniform sampling.
    
    Strategy:
    - Always include first and last frame
    - Uniformly sample (k-2) frames in between
    
    Args:
        num_frames: Total number of frames in video
        k: Number of keyframes to select
        
    Returns:
        List of frame indices (sorted)
    """
    if num_frames <= 0:
        return []
    
    if k >= num_frames:
        # If k >= num_frames, use all frames as keyframes
        return list(range(num_frames))
    
    if k <= 2:
        # Minimum: first and last frame
        return [0, num_frames - 1] if num_frames > 1 else [0]
    
    # Always include first and last
    keyframes = [0, num_frames - 1]
    
    # Uniformly sample (k-2) frames in between
    # Use linspace to get evenly distributed indices
    if k > 2:
        middle_frames = np.linspace(1, num_frames - 2, k - 2, dtype=int)
        keyframes.extend(middle_frames.tolist())
    
    # Sort and remove duplicates
    keyframes = sorted(set(keyframes))
    
    logger.info(f"Selected {len(keyframes)} keyframes from {num_frames} frames: {keyframes[:5]}...")
    
    return keyframes


def interpolate_masks(
    keyframe_masks: Dict[int, np.ndarray],
    keyframe_indices: List[int],
    total_frames: int
) -> Dict[int, np.ndarray]:
    """
    Interpolate masks for non-keyframe frames.
    
    Strategy (simple nearest-neighbor):
    - For each non-keyframe frame, use the mask from the closest keyframe
    - Can be enhanced later with linear interpolation or optical flow
    
    Args:
        keyframe_masks: Dict of keyframe_idx -> mask array
        keyframe_indices: List of keyframe indices
        total_frames: Total number of frames to generate masks for
        
    Returns:
        Dict of frame_idx -> mask for all frames
    """
    if not keyframe_masks or not keyframe_indices:
        return {}
    
    # Sort keyframe indices
    keyframe_indices = sorted(keyframe_indices)
    
    # Initialize output with keyframe masks
    all_masks = {idx: mask for idx, mask in keyframe_masks.items()}
    
    # Interpolate for non-keyframe frames
    for frame_idx in range(total_frames):
        if frame_idx in all_masks:
            continue  # Already have mask for this keyframe
        
        # Find closest keyframe (nearest neighbor)
        closest_keyframe = min(
            keyframe_indices,
            key=lambda k: abs(k - frame_idx)
        )
        
        # Copy mask from closest keyframe
        if closest_keyframe in keyframe_masks:
            all_masks[frame_idx] = keyframe_masks[closest_keyframe].copy()
    
    logger.info(f"Interpolated masks: {len(keyframe_masks)} keyframes -> {len(all_masks)} total frames")
    
    return all_masks


def get_processing_params(mode: str) -> Tuple[float, int]:
    """
    Get processing parameters based on mode.
    
    Args:
        mode: "fast" or "accurate"
        
    Returns:
        Tuple of (target_fps, num_keyframes)
    """
    # Import here to avoid circular dependency
    from config import (
        PROCESSING_FPS_FAST,
        PROCESSING_FPS_ACCURATE,
        KEYFRAMES_FAST,
        KEYFRAMES_ACCURATE
    )
    
    if mode.lower() == "accurate":
        return PROCESSING_FPS_ACCURATE, KEYFRAMES_ACCURATE
    else:
        # Default to fast mode
        return PROCESSING_FPS_FAST, KEYFRAMES_FAST


def cleanup_temp_video(video_path: str) -> None:
    """
    Clean up temporary downsampled video file.
    
    Args:
        video_path: Path to temporary video file
    """
    try:
        path = Path(video_path)
        if path.exists() and path.parent.name == tempfile.gettempdir():
            path.unlink()
            logger.debug(f"Cleaned up temp video: {video_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp video {video_path}: {e}")
