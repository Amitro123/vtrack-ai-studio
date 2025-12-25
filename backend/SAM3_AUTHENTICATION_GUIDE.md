# SAM3 Authentication Guide

## Current Status ✅
- ✅ SAM3 repository cloned and installed
- ✅ SAM3 Python package installed in `sam3` environment
- ✅ Hugging Face Hub CLI installed
- ⚠️ **SAM3 checkpoints not downloaded** (requires authentication)

## Next Steps to Enable SAM3

### Step 1: Request Access to SAM3 Model
1. Visit: https://huggingface.co/facebook/sam3
2. Click "Request Access" button
3. Wait for approval (usually instant or within a few hours)

### Step 2: Create Hugging Face Access Token
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "SAM3 Access")
4. Select "Read" permission
5. Click "Generate token"
6. **Copy the token** (you won't be able to see it again!)

### Step 3: Authenticate with Hugging Face
Open PowerShell and run:
```powershell
# Activate the sam3 environment
conda activate sam3

# Login to Hugging Face
huggingface-cli login

# When prompted, paste your access token
# Press Enter to confirm
```

### Step 4: Download SAM3 Checkpoint
After authentication, you can download the checkpoint using Python:

```powershell
# Activate environment
conda activate sam3

# Navigate to backend
cd backend

# Run Python script to download checkpoint
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam3', filename='sam3_hiera_large.pt', local_dir='checkpoints/sam3')"
```

**Alternative checkpoints:**
- `sam3_hiera_large.pt` (~2.5GB) - Recommended balance
- `sam3_hiera_huge.pt` (~3.5GB) - Best quality
- `sam3_hiera_base_plus.pt` (~1.5GB) - Faster, lower quality

### Step 5: Verify Installation
```powershell
# Check if SAM3 is working
python -c "from sam3 import build_sam3_video_predictor; print('SAM3 ready!')"

# Check health endpoint
Invoke-RestMethod -Uri http://localhost:8000/api/health -Method Get | ConvertTo-Json
```

Expected output:
```json
{
  "status": "healthy",
  "sam3_available": true,
  "sam3_setup_valid": true,
  "device": "cpu",
  ...
}
```

## Alternative: Run Without SAM3

If you want to test the application without SAM3 video segmentation, you can:

1. **Use the backend without SAM3** - The backend will work for other features (audio processing, etc.)
2. **Mock SAM3 functionality** - For development/testing purposes

The backend is already running and healthy, just without SAM3 capabilities.

## Troubleshooting

### "Access denied" when downloading
- Make sure you've requested access on the Hugging Face page
- Wait for approval (check your email)
- Verify you're logged in: `huggingface-cli whoami`

### "Token invalid"
- Generate a new token with "Read" permissions
- Run `huggingface-cli login` again

### "Checkpoint not found"
- Check the checkpoint filename matches exactly
- Verify the file was downloaded to `backend/checkpoints/sam3/`
- List files: `ls backend/checkpoints/sam3/`

## Current Backend Status

Your backend is running successfully at http://localhost:8000 with:
- ✅ FastAPI server
- ✅ PyTorch 2.9.1 with CUDA 12.6 binaries
- ✅ Audio processing (demucs)
- ⚠️ SAM3 video segmentation (pending checkpoint download)

You can still use the application for non-video features!
