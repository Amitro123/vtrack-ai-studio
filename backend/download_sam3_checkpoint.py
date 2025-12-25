"""
SAM3 Checkpoint Downloader
Downloads SAM3 model checkpoint from Hugging Face
Requires: Hugging Face authentication (run hf_login.py first)
"""

from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

def download_checkpoint(model_size="large"):
    """
    Download SAM3 checkpoint from Hugging Face
    
    Args:
        model_size: "large" (default, 2.5GB), "huge" (3.5GB), or "base_plus" (1.5GB)
    """
    
    # Checkpoint filenames
    checkpoints = {
        "large": "sam3_hiera_large.pt",
        "huge": "sam3_hiera_huge.pt",
        "base_plus": "sam3_hiera_base_plus.pt"
    }
    
    if model_size not in checkpoints:
        print(f"❌ Invalid model size: {model_size}")
        print(f"   Available: {', '.join(checkpoints.keys())}")
        sys.exit(1)
    
    filename = checkpoints[model_size]
    repo_id = "facebook/sam3"
    local_dir = Path("checkpoints/sam3")
    
    print("=" * 60)
    print("SAM3 Checkpoint Downloader")
    print("=" * 60)
    print()
    print(f"📦 Model: {model_size}")
    print(f"📁 File: {filename}")
    print(f"🏠 Repo: {repo_id}")
    print(f"💾 Destination: {local_dir}")
    print()
    
    # Create directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("⬇️  Downloading checkpoint...")
        print("   (This may take several minutes depending on your connection)")
        print()
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        print()
        print("✅ Download complete!")
        print(f"📍 Checkpoint saved to: {downloaded_path}")
        print()
        print("🎉 SAM3 is now ready to use!")
        print("   Restart the backend server to load the checkpoint.")
        
    except Exception as e:
        print()
        print(f"❌ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you've requested access at: https://huggingface.co/facebook/sam3")
        print("2. Run 'python hf_login.py' to authenticate")
        print("3. Wait for access approval (usually instant)")
        sys.exit(1)

if __name__ == "__main__":
    # Default to large model
    model_size = sys.argv[1] if len(sys.argv) > 1 else "large"
    download_checkpoint(model_size)
