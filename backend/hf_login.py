"""
Hugging Face Authentication Helper (Environment Variable Version)
Authenticate with Hugging Face using token from environment or .env file
"""

from huggingface_hub import login
import sys
import os
from pathlib import Path

def authenticate():
    """Login to Hugging Face using token from environment"""
    print("=" * 60)
    print("Hugging Face Authentication")
    print("=" * 60)
    print()
    
    # Try to get token from environment variable
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_TOKEN')
    
    # If not in env, try to read from .env file
    if not token:
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            print(f"📄 Reading token from: {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN=') or line.startswith('HUGGING_FACE_TOKEN='):
                        token = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
    
    if not token:
        print("❌ No token found!")
        print()
        print("Please add your Hugging Face token to .env file:")
        print("  HF_TOKEN=your_token_here")
        print()
        print("Or set environment variable:")
        print("  $env:HF_TOKEN='your_token_here'  # PowerShell")
        print()
        sys.exit(1)
    
    try:
        # Login with the token
        print("🔐 Authenticating with Hugging Face...")
        login(token=token, add_to_git_credential=True)
        
        print("\n✅ Successfully authenticated with Hugging Face!")
        print("You can now download SAM3 checkpoints.")
        
        # Verify authentication
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"\n👤 Logged in as: {user_info['name']}")
        print()
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure your token is valid")
        print("2. Request access at: https://huggingface.co/facebook/sam3")
        print("3. Create token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

if __name__ == "__main__":
    authenticate()
