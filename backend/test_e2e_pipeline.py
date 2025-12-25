"""
End-to-end tests for the complete video processing pipeline
These tests verify the full flow from video upload to processed output
"""

import pytest
from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
from unittest.mock import patch, Mock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from video_preprocessor import (
    downsample_video,
    select_keyframes,
    interpolate_masks,
    get_processing_params
)


class TestVideoPreprocessingPipeline:
    """Test the complete video preprocessing pipeline"""
    
    @pytest.fixture
    def sample_video(self):
        """Create a sample test video"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Create a simple test video (10 frames, 640x480, 25 fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 25.0, (640, 480))
        
        for i in range(10):
            # Create a frame with different color for each frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 25)
            out.write(frame)
        
        out.release()
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_full_preprocessing_pipeline_fast_mode(self, sample_video):
        """Test complete preprocessing pipeline in fast mode"""
        # Get fast mode parameters
        target_fps, num_keyframes = get_processing_params("fast")
        
        # Step 1: Downsample video
        downsampled_path, frame_mapping = downsample_video(sample_video, target_fps)
        
        assert downsampled_path != sample_video  # Should create new file
        assert len(frame_mapping) > 0
        assert 0 in frame_mapping  # First frame should be mapped
        
        # Step 2: Select keyframes
        keyframe_indices = select_keyframes(len(frame_mapping), num_keyframes)
        
        assert len(keyframe_indices) <= num_keyframes
        assert 0 in keyframe_indices  # First frame
        assert max(keyframe_indices) == len(frame_mapping) - 1  # Last frame
        
        # Step 3: Create dummy masks for keyframes
        keyframe_masks = {}
        for idx in keyframe_indices:
            keyframe_masks[idx] = np.ones((480, 640), dtype=bool)
        
        # Step 4: Interpolate masks
        all_masks = interpolate_masks(keyframe_masks, keyframe_indices, len(frame_mapping))
        
        assert len(all_masks) == len(frame_mapping)
        assert all(idx in all_masks for idx in keyframe_indices)
        
        # Cleanup downsampled video
        Path(downsampled_path).unlink(missing_ok=True)
    
    def test_full_preprocessing_pipeline_accurate_mode(self, sample_video):
        """Test complete preprocessing pipeline in accurate mode"""
        # Get accurate mode parameters
        target_fps, num_keyframes = get_processing_params("accurate")
        
        # Run full pipeline
        downsampled_path, frame_mapping = downsample_video(sample_video, target_fps)
        keyframe_indices = select_keyframes(len(frame_mapping), num_keyframes)
        
        # Accurate mode should have more keyframes
        assert len(keyframe_indices) > 12  # More than fast mode
        
        # Cleanup
        if downsampled_path != sample_video:
            Path(downsampled_path).unlink(missing_ok=True)
    
    def test_no_downsampling_when_not_needed(self, sample_video):
        """Test that no downsampling occurs when target FPS >= original FPS"""
        # Original video is 25 FPS, request 30 FPS
        downsampled_path, frame_mapping = downsample_video(sample_video, 30.0)
        
        # Should return original video path
        assert downsampled_path == sample_video
        
        # Frame mapping should be identity
        assert all(frame_mapping[i] == i for i in frame_mapping.keys())


class TestMaskInterpolationQuality:
    """Test mask interpolation quality and correctness"""
    
    def test_interpolation_preserves_keyframe_masks(self):
        """Test that keyframe masks are preserved exactly"""
        keyframe_masks = {
            0: np.zeros((10, 10), dtype=bool),
            5: np.ones((10, 10), dtype=bool),
            10: np.zeros((10, 10), dtype=bool)
        }
        keyframe_indices = [0, 5, 10]
        
        all_masks = interpolate_masks(keyframe_masks, keyframe_indices, 11)
        
        # Keyframe masks should be identical
        assert np.array_equal(all_masks[0], keyframe_masks[0])
        assert np.array_equal(all_masks[5], keyframe_masks[5])
        assert np.array_equal(all_masks[10], keyframe_masks[10])
    
    def test_interpolation_nearest_neighbor_logic(self):
        """Test that interpolation uses nearest neighbor correctly"""
        keyframe_masks = {
            0: np.zeros((5, 5), dtype=bool),
            10: np.ones((5, 5), dtype=bool)
        }
        keyframe_indices = [0, 10]
        
        all_masks = interpolate_masks(keyframe_masks, keyframe_indices, 11)
        
        # Frames 0-5 should be closer to frame 0 (zeros)
        for i in range(0, 6):
            assert np.array_equal(all_masks[i], keyframe_masks[0])
        
        # Frames 6-10 should be closer to frame 10 (ones)
        for i in range(6, 11):
            assert np.array_equal(all_masks[i], keyframe_masks[10])


class TestProcessingModeConsistency:
    """Test that processing modes are consistent throughout the pipeline"""
    
    def test_fast_mode_parameters(self):
        """Test fast mode returns correct parameters"""
        fps, keyframes = get_processing_params("fast")
        
        from config import PROCESSING_FPS_FAST, KEYFRAMES_FAST
        assert fps == PROCESSING_FPS_FAST
        assert keyframes == KEYFRAMES_FAST
    
    def test_accurate_mode_parameters(self):
        """Test accurate mode returns correct parameters"""
        fps, keyframes = get_processing_params("accurate")
        
        from config import PROCESSING_FPS_ACCURATE, KEYFRAMES_ACCURATE
        assert fps == PROCESSING_FPS_ACCURATE
        assert keyframes == KEYFRAMES_ACCURATE
    
    def test_invalid_mode_defaults_to_fast(self):
        """Test that invalid mode defaults to fast"""
        fps, keyframes = get_processing_params("invalid_mode")
        
        from config import PROCESSING_FPS_FAST, KEYFRAMES_FAST
        assert fps == PROCESSING_FPS_FAST
        assert keyframes == KEYFRAMES_FAST


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_frame_video(self):
        """Test handling of single-frame video"""
        keyframes = select_keyframes(1, 10)
        assert len(keyframes) == 1
        assert keyframes == [0]
    
    def test_more_keyframes_than_frames(self):
        """Test when requested keyframes > available frames"""
        keyframes = select_keyframes(5, 20)
        assert len(keyframes) == 5
        assert keyframes == [0, 1, 2, 3, 4]
    
    def test_empty_keyframe_masks(self):
        """Test interpolation with empty keyframe masks"""
        all_masks = interpolate_masks({}, [], 10)
        assert len(all_masks) == 0
    
    def test_zero_frames(self):
        """Test handling of zero frames"""
        keyframes = select_keyframes(0, 10)
        assert len(keyframes) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
