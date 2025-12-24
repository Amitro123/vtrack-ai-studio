"""
VTrackAI Core Integration
Imports and wraps VTrackAI core modules for use in the backend.
"""

import sys
from pathlib import Path

# Add VTrackAI to Python path
VTRACK_AI_PATH = Path(__file__).parent.parent.parent / "VTrackAI"
sys.path.insert(0, str(VTRACK_AI_PATH))

# Import VTrackAI core modules
try:
    from core.sam2_tracker import SAM2Tracker
    from core.sam_audio import SAMAudio
    from core.grounding_dino import GroundingDINO
    from core.propainter import ProPainter
    from utils import video_utils, audio_utils
    import config as vtrack_config
    
    print("✅ VTrackAI core modules loaded successfully")
    VTRACK_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  VTrackAI core modules not available: {e}")
    print(f"   Looking in: {VTRACK_AI_PATH}")
    VTRACK_AVAILABLE = False
    
    # Fallback: create dummy classes for development
    class SAM2Tracker:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VTrackAI not available")
    
    class SAMAudio:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VTrackAI not available")
    
    class GroundingDINO:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VTrackAI not available")
    
    class ProPainter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("VTrackAI not available")


class VTrackAICore:
    """
    Singleton wrapper for VTrackAI core modules.
    Lazy-loads models on demand to save memory.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.sam2_tracker = None
        self.sam_audio = None
        self.grounding_dino = None
        self.propainter = None
        self._initialized = True
    
    def get_sam2_tracker(self) -> SAM2Tracker:
        """Lazy load SAM2 tracker."""
        if self.sam2_tracker is None:
            print("Loading SAM 2 tracker...")
            self.sam2_tracker = SAM2Tracker()
        return self.sam2_tracker
    
    def get_sam_audio(self) -> SAMAudio:
        """Lazy load SAM Audio."""
        if self.sam_audio is None:
            print("Loading SAM Audio...")
            self.sam_audio = SAMAudio()
        return self.sam_audio
    
    def get_grounding_dino(self) -> GroundingDINO:
        """Lazy load Grounding DINO."""
        if self.grounding_dino is None:
            print("Loading Grounding DINO...")
            self.grounding_dino = GroundingDINO()
        return self.grounding_dino
    
    def get_propainter(self) -> ProPainter:
        """Lazy load ProPainter."""
        if self.propainter is None:
            print("Loading ProPainter...")
            self.propainter = ProPainter()
        return self.propainter
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.sam2_tracker:
            self.sam2_tracker.clear_cache()
        if self.sam_audio:
            self.sam_audio.clear_cache()
        if self.grounding_dino:
            self.grounding_dino.clear_cache()
        if self.propainter:
            self.propainter.clear_cache()


# Global instance
vtrack_core = VTrackAICore()
