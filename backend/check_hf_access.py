"""
Check Hugging Face Access to SAM3
Verifies if you have access to the facebook/sam3 repository
"""

from huggingface_hub import whoami, repo_info
import sys

def check_access():
    """Check if user has access to SAM3 repository"""
    print("=" * 60)
    print("Checking Hugging Face Access to SAM3")
    print("=" * 60)
    print()
    
    try:
        # Check if logged in
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        print()
        
        # Try to access the SAM3 repo
        repo_id = "facebook/sam3"
        print(f"🔍 Checking access to: {repo_id}")
        
        try:
            info = repo_info(repo_id, repo_type="model")
            print(f"✅ You have access to {repo_id}!")
            print()
            print("📦 Available files:")
            for sibling in info.siblings[:10]:  # Show first 10 files
                size_mb = sibling.size / (1024 * 1024) if sibling.size else 0
                print(f"   - {sibling.rfilename} ({size_mb:.1f} MB)")
            print()
            print("🎉 You're ready to download SAM3 checkpoints!")
            
        except Exception as e:
            print(f"❌ No access to {repo_id}")
            print()
            print(f"Error: {e}")
            print()
            print("📝 To request access:")
            print(f"   1. Visit: https://huggingface.co/{repo_id}")
            print("   2. Click 'Request Access' button")
            print("   3. Wait for approval (usually instant)")
            print("   4. Run this script again to verify")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        print()
        print("Run: python hf_login.py")
        sys.exit(1)

if __name__ == "__main__":
    check_access()
