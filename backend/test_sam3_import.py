"""
Test SAM3 import and basic functionality
"""
import sys
from pathlib import Path

# Add SAM3 to path
SAM3_PATH = Path(__file__).parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

print("Testing SAM3 import...")
try:
    from sam3 import build_sam3_video_predictor
    print("✅ SAM3 import successful")
    print(f"   build_sam3_video_predictor: {build_sam3_video_predictor}")
    
    # Try to build predictor
    print("\nTesting SAM3 predictor build...")
    checkpoint = Path("checkpoints/sam3/sam3.pt")
    if checkpoint.exists():
        print(f"✅ Checkpoint found: {checkpoint}")
        try:
            predictor = build_sam3_video_predictor(
                ckpt_path=str(checkpoint),
                device="cpu"
            )
            print(f"✅ Predictor created: {type(predictor)}")
            print(f"   Available methods: {[m for m in dir(predictor) if not m.startswith('_')]}")
        except Exception as e:
            print(f"❌ Failed to create predictor: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Checkpoint not found: {checkpoint}")
        
except ImportError as e:
    print(f"❌ SAM3 import failed: {e}")
    import traceback
    traceback.print_exc()
