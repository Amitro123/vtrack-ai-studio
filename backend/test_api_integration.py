"""
Integration tests for FastAPI endpoints
Tests API endpoints with mocked SAM3 engine
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import io
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from server import app
import config


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_video_file():
    """Create a mock video file for testing"""
    # Create a minimal valid video file content
    video_content = b"fake video content for testing"
    return ("test_video.mp4", io.BytesIO(video_content), "video/mp4")


@pytest.fixture
def mock_sam3_engine():
    """Mock SAM3 engine with realistic responses"""
    with patch('server.get_engine') as mock_get_engine:
        mock_engine = Mock()
        
        # Mock track_from_point response
        mock_engine.track_from_point.return_value = {
            "masked_video_path": "/tmp/masked_video.mp4",
            "masks": {0: np.ones((10, 10), dtype=bool)},
            "tracks": [{"obj_id": 1, "frames": [0]}],
            "metadata": {
                "fps": 25.0,
                "resolution": (640, 480),
                "total_frames": 50,
                "tracked_frames": 50,
                "processing_mode": "fast",
                "keyframes_used": 12,
                "processed_fps": 5.0
            }
        }
        
        # Mock track_from_text response
        mock_engine.track_from_text.return_value = {
            "highlighted_video_path": "/tmp/highlighted_video.mp4",
            "masks": {0: np.ones((10, 10), dtype=bool)},
            "tracks": [{"obj_id": 1, "frames": [0], "text": "test"}],
            "detected_objects": [{"obj_id": 1, "text": "test", "confidence": 0.95}],
            "metadata": {
                "fps": 25.0,
                "resolution": (640, 480),
                "total_frames": 50,
                "tracked_frames": 50,
                "prompt": "test",
                "processing_mode": "fast",
                "keyframes_used": 12,
                "processed_fps": 5.0
            }
        }
        
        # Mock segment_for_removal response
        mock_engine.segment_for_removal.return_value = {
            "masks": {0: np.ones((10, 10), dtype=bool)},
            "tracks": [{"obj_id": 1, "frames": [0]}]
        }
        
        mock_get_engine.return_value = mock_engine
        yield mock_engine


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "sam3_available" in data
    
    def test_health_endpoint(self, client):
        """Test detailed health endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "sam3_available" in data
        assert "device" in data
        assert "active_tasks" in data


class TestTrackPointEndpoint:
    """Test /api/track-point endpoint"""
    
    @patch('server.validate_video')
    @patch('server.shutil.move')
    def test_track_point_fast_mode(self, mock_move, mock_validate, client, mock_video_file, mock_sam3_engine):
        """Test track-point with fast mode"""
        mock_validate.return_value = (True, "Valid")
        
        response = client.post(
            "/api/track-point",
            files={"video": mock_video_file},
            data={"x": 50.0, "y": 50.0, "frame_idx": 0, "mode": "fast"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response format
        assert "task_id" in data
        assert data["mode"] == "fast"
        assert data["processing_fps"] == config.PROCESSING_FPS_FAST
        assert data["num_keyframes"] == config.KEYFRAMES_FAST
        assert "masked_video_url" in data
        assert "processing_steps" in data
        
        # Verify SAM3 engine was called with correct mode
        mock_sam3_engine.track_from_point.assert_called_once()
        call_kwargs = mock_sam3_engine.track_from_point.call_args[1]
        assert call_kwargs["mode"] == "fast"
    
    @patch('server.validate_video')
    @patch('server.shutil.move')
    def test_track_point_accurate_mode(self, mock_move, mock_validate, client, mock_video_file, mock_sam3_engine):
        """Test track-point with accurate mode"""
        mock_validate.return_value = (True, "Valid")
        
        response = client.post(
            "/api/track-point",
            files={"video": mock_video_file},
            data={"x": 50.0, "y": 50.0, "frame_idx": 0, "mode": "accurate"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "accurate"
        assert data["processing_fps"] == config.PROCESSING_FPS_ACCURATE
        assert data["num_keyframes"] == config.KEYFRAMES_ACCURATE
    
    def test_track_point_invalid_mode(self, client, mock_video_file):
        """Test track-point with invalid mode"""
        response = client.post(
            "/api/track-point",
            files={"video": mock_video_file},
            data={"x": 50.0, "y": 50.0, "frame_idx": 0, "mode": "invalid"}
        )
        
        assert response.status_code == 400
        assert "Mode must be 'fast' or 'accurate'" in response.json()["detail"]
    
    def test_track_point_missing_parameters(self, client, mock_video_file):
        """Test track-point with missing parameters"""
        response = client.post(
            "/api/track-point",
            files={"video": mock_video_file},
            data={"x": 50.0}  # Missing y
        )
        
        assert response.status_code == 422  # Validation error


class TestTextToVideoEndpoint:
    """Test /api/text-to-video endpoint"""
    
    @patch('server.validate_video')
    @patch('server.shutil.move')
    @patch('server.subprocess.run')
    def test_text_to_video_fast_mode(self, mock_subprocess, mock_move, mock_validate, 
                                     client, mock_video_file, mock_sam3_engine):
        """Test text-to-video with fast mode"""
        mock_validate.return_value = (True, "Valid")
        
        response = client.post(
            "/api/text-to-video",
            files={"video": mock_video_file},
            data={"prompt": "isolate drums", "mode": "fast"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "fast"
        assert data["processing_fps"] == config.PROCESSING_FPS_FAST
        assert data["num_keyframes"] == config.KEYFRAMES_FAST
        assert "highlighted_video_url" in data
        assert "audio_url" in data
        assert "ai_response" in data
    
    @patch('server.validate_video')
    @patch('server.shutil.move')
    @patch('server.subprocess.run')
    def test_text_to_video_accurate_mode(self, mock_subprocess, mock_move, mock_validate,
                                         client, mock_video_file, mock_sam3_engine):
        """Test text-to-video with accurate mode"""
        mock_validate.return_value = (True, "Valid")
        
        response = client.post(
            "/api/text-to-video",
            files={"video": mock_video_file},
            data={"prompt": "isolate vocals", "mode": "accurate"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "accurate"
        assert data["processing_fps"] == config.PROCESSING_FPS_ACCURATE
        assert data["num_keyframes"] == config.KEYFRAMES_ACCURATE


class TestRemoveObjectEndpoint:
    """Test /api/remove-object endpoint"""
    
    @patch('server.validate_video')
    @patch('server.cv2.VideoCapture')
    @patch('server.cv2.VideoWriter')
    def test_remove_object_fast_mode(self, mock_writer, mock_capture, mock_validate,
                                     client, mock_video_file, mock_sam3_engine):
        """Test remove-object with fast mode"""
        mock_validate.return_value = (True, "Valid")
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 640, 480]  # fps, width, height
        mock_cap.read.return_value = (False, None)  # No frames to read
        mock_capture.return_value = mock_cap
        
        response = client.post(
            "/api/remove-object",
            files={"video": mock_video_file},
            data={"x": 50.0, "y": 50.0, "frame_idx": 0, "mode": "fast"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "fast"
        assert data["processing_fps"] == config.PROCESSING_FPS_FAST
        assert data["num_keyframes"] == config.KEYFRAMES_FAST
        assert "inpainted_video_url" in data


class TestTaskStatusEndpoint:
    """Test task status endpoint"""
    
    def test_get_task_status_not_found(self, client):
        """Test getting status of non-existent task"""
        response = client.get("/api/task/nonexistent-task-id")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
