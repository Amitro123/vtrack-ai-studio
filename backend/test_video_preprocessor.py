"""
Tests for video_preprocessor module
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from video_preprocessor import (
    select_keyframes,
    interpolate_masks,
    get_processing_params
)


class TestKeyframeSelection:
    """Test keyframe selection logic"""
    
    def test_select_keyframes_basic(self):
        """Test basic keyframe selection"""
        # 50 frames, select 10 keyframes
        keyframes = select_keyframes(50, 10)
        
        assert len(keyframes) == 10
        assert keyframes[0] == 0  # First frame
        assert keyframes[-1] == 49  # Last frame
        assert keyframes == sorted(keyframes)  # Should be sorted
    
    def test_select_keyframes_all_frames(self):
        """Test when k >= num_frames"""
        keyframes = select_keyframes(10, 15)
        
        assert len(keyframes) == 10
        assert keyframes == list(range(10))
    
    def test_select_keyframes_minimum(self):
        """Test minimum keyframes (first and last)"""
        keyframes = select_keyframes(100, 2)
        
        assert len(keyframes) == 2
        assert keyframes == [0, 99]
    
    def test_select_keyframes_single_frame(self):
        """Test single frame video"""
        keyframes = select_keyframes(1, 10)
        
        assert len(keyframes) == 1
        assert keyframes == [0]


class TestMaskInterpolation:
    """Test mask interpolation logic"""
    
    def test_interpolate_masks_basic(self):
        """Test basic mask interpolation"""
        # Create dummy masks for keyframes 0, 25, 50
        keyframe_masks = {
            0: np.ones((10, 10), dtype=bool),
            25: np.ones((10, 10), dtype=bool),
            50: np.ones((10, 10), dtype=bool)
        }
        keyframe_indices = [0, 25, 50]
        
        all_masks = interpolate_masks(keyframe_masks, keyframe_indices, 51)
        
        assert len(all_masks) == 51
        assert 0 in all_masks
        assert 25 in all_masks
        assert 50 in all_masks
        assert 10 in all_masks  # Interpolated frame
    
    def test_interpolate_masks_nearest_neighbor(self):
        """Test that interpolation uses nearest neighbor"""
        keyframe_masks = {
            0: np.zeros((5, 5), dtype=bool),
            10: np.ones((5, 5), dtype=bool)
        }
        keyframe_indices = [0, 10]
        
        all_masks = interpolate_masks(keyframe_masks, keyframe_indices, 11)
        
        # Frame 4 should be closer to 0, so should be zeros
        assert np.array_equal(all_masks[4], keyframe_masks[0])
        
        # Frame 6 should be closer to 10, so should be ones
        assert np.array_equal(all_masks[6], keyframe_masks[10])


class TestProcessingParams:
    """Test processing parameter retrieval"""
    
    def test_get_processing_params_fast(self):
        """Test fast mode parameters"""
        fps, keyframes = get_processing_params("fast")
        
        assert fps == 5.0
        assert keyframes == 12
    
    def test_get_processing_params_accurate(self):
        """Test accurate mode parameters"""
        fps, keyframes = get_processing_params("accurate")
        
        assert fps == 10.0
        assert keyframes == 32
    
    def test_get_processing_params_default(self):
        """Test default mode (should be fast)"""
        fps, keyframes = get_processing_params("invalid")
        
        # Should default to fast mode
        assert fps == 5.0
        assert keyframes == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
