# SAM3 Setup Guide

## Prerequisites

- **Python** 3.12
- **CUDA** 12.6 or higher
- **conda** (recommended for environment management)
- **GPU** with 16GB+ VRAM (T4, A10G, A100, etc.)

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n sam3 python=3.12
conda activate sam3
```

### 2. Install PyTorch with CUDA 12.6

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Clone and Install SAM3

```bash
cd backend
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

### 4. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download SAM3 Checkpoint

```bash
# Create checkpoints directory
mkdir -p checkpoints/sam3

# Download SAM3 Large checkpoint (~2.5GB)
cd checkpoints/sam3
wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt
cd ../..
```

**Alternative checkpoints**:
- SAM3 Huge: `sam3_hiera_huge.pt` (~3.5GB, best quality)
- SAM3 Base+: `sam3_hiera_base_plus.pt` (~1.5GB, faster)

### 6. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from sam3 import build_sam3_video_predictor; print('SAM3 installed successfully')"
```

### 7. Start Backend

```bash
python server.py
```

Expected output:
```
============================================================
🚀 VTrackAI Studio Backend Server (SAM3)
============================================================
✅ SAM3 setup valid (device: cuda, VRAM: 16GB)
Upload Directory: .../backend/uploads
Server: http://0.0.0.0:8000
============================================================
```

## Troubleshooting

### "SAM3 not found"

```bash
cd backend
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### "SAM3 checkpoint not found"

```bash
mkdir -p backend/checkpoints/sam3
wget -O backend/checkpoints/sam3/sam3_hiera_large.pt \
  https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt
```

### "CUDA out of memory"

- Use a smaller checkpoint (Base+ instead of Large)
- Reduce video resolution to 360p
- Process shorter videos (<5s)
- Enable half precision in config.py

### "torch not found" or version mismatch

```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Google Colab Setup

```python
# Install PyTorch with CUDA
!pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone SAM3
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install -e .
%cd ..

# Download checkpoint
!mkdir -p checkpoints/sam3
!wget -O checkpoints/sam3/sam3_hiera_large.pt \
  https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt

# Install backend dependencies
!pip install -r backend/requirements.txt

# Start server
%cd backend
!python server.py
```

## Performance Tips

1. **Use half precision** (FP16) for faster inference:
   ```python
   # In config.py
   USE_HALF_PRECISION = True
   ```

2. **Batch processing**: Process multiple videos sequentially to reuse loaded model

3. **GPU selection**: Use the best available GPU
   ```bash
   CUDA_VISIBLE_DEVICES=0 python server.py
   ```

4. **Memory management**: Clear cache between requests
   ```python
   torch.cuda.empty_cache()
   ```

## Next Steps

After setup:
1. Start the frontend: `npm run dev`
2. Open http://localhost:5173
3. Test all three workflows:
   - Point-to-Track
   - Text-to-Video
   - Object Removal
