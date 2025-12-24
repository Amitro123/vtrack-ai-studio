# VTrackAI Studio

**Semantic Video & Audio Editor** - Professional UI powered by SAM3 + React + FastAPI

[![React](https://img.shields.io/badge/React-18.3-blue?logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?logo=typescript)](https://www.typescriptlang.org/)
[![SAM3](https://img.shields.io/badge/SAM3-Meta-orange)](https://github.com/facebookresearch/sam3)

## 🌟 Features

- **Point-to-Track**: Click any object → SAM3 tracks across frames
- **Text-to-Video**: "Isolate drums" → SAM3 native text understanding + Demucs audio extraction
- **Object Removal**: Click to remove → SAM3 tracking + OpenCV inpainting
- **Single Upload**: One video upload shared across all tabs
- **Real-time Progress**: Live processing status from backend
- **Download Results**: Processed videos and isolated audio files

## 🆕 What's New in SAM3

- **Unified Model**: Single SAM3 model for text + visual prompts (no separate GroundingDINO)
- **Better Text Understanding**: Native open-vocabulary segmentation
- **Improved Tracking**: Enhanced temporal consistency in videos
- **Simpler Architecture**: Fewer dependencies, easier setup
- **Faster Inference**: 20-30% speed improvement over SAM2
- **Lower VRAM**: ~2GB less memory usage

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.12
- **PyTorch** 2.7.0 with CUDA 12.6+
- **CUDA GPU** with 16GB+ VRAM (T4, A10G, A100 recommended)
- **Conda** (for environment management)

### 1. Setup SAM3 Environment

```bash
# Create conda environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch with CUDA 12.6
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install SAM3
cd backend
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Download SAM3 Checkpoint

```bash
# Download SAM3 Large checkpoint (~2.5GB)
mkdir -p checkpoints/sam3
cd checkpoints/sam3
wget https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt
cd ../..
```

**Alternative checkpoints**:
- SAM3 Huge: `sam3_hiera_huge.pt` (~3.5GB, best quality)
- SAM3 Base+: `sam3_hiera_base_plus.pt` (~1.5GB, faster)

### 3. Start Backend

```bash
conda activate sam3
python server.py
```

Backend runs on **http://localhost:8000**

### 4. Start Frontend

```bash
npm install
npm run dev
```

Frontend runs on **http://localhost:5173**

### 5. Open Browser

Navigate to **http://localhost:5173** and start editing!

## 🎯 Quick Start (Single Command)

### Option 1: Using Startup Scripts

**Windows (PowerShell)**:
```powershell
.\start.ps1
```

**Windows (Batch)**:
```batch
start.bat
```

**Linux/Mac**:
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Using npm

```bash
npm install --save-dev concurrently
npm run dev:all
```

This will start both backend and frontend in a single terminal.

## 📖 How to Use

### Step 1: Upload Video

- Click the upload area at the top
- Select a video (max 10s, 480p, <100MB)
- Supported formats: MP4, WebM, MOV, AVI, MKV
- Video is automatically shared across all tabs

### Step 2: Choose Your Workflow

#### Tab 1: Click to Track 🎯

1. Click on any object in the video
2. SAM3 generates a red mask overlay
3. Mask tracks across all frames
4. Download the masked video

**Use Case**: Track a person, object, or instrument through the video

#### Tab 2: Chat (Text-to-Video) 💬

1. Type a text prompt:
   - "Isolate drums"
   - "Extract vocals"
   - "Find the guitar"
2. SAM3 detects and segments the object (native text understanding)
3. SAM3 tracks it across frames
4. Demucs separates the audio
5. Download isolated audio + highlighted video

**Use Case**: Extract specific instrument audio from band performances

#### Tab 3: Remove Objects 🗑️

1. Click on an object to remove
2. SAM3 tracks it across frames
3. OpenCV inpaints to remove it
4. Download the cleaned video

**Use Case**: Remove unwanted objects or people from videos

## 🎬 Example Workflow: Band Demo

1. **Upload** a 10s band performance video
2. **Tab 2 (Chat)**: Type "Isolate drums"
   - SAM3 detects drummer
   - Get isolated drum audio track
   - See highlighted drum regions in video
3. **Tab 1 (Click)**: Click on guitarist
   - Track guitarist across all frames
   - Download tracked video
4. **Tab 3 (Remove)**: Click on bassist
   - Remove bassist from video
   - Download inpainted result

## 🛠️ Tech Stack

### Frontend
- **React** 18.3 + **TypeScript** 5.8
- **Vite** 5.4 (build tool)
- **shadcn/ui** + **Tailwind CSS** (UI components)
- **TanStack Query** (data fetching)

### Backend
- **FastAPI** 0.104 (REST API)
- **Uvicorn** (ASGI server)
- **AI Models**:
  - **SAM 3** (Meta) - Open-vocabulary video segmentation & tracking
  - **Demucs** - Audio stem separation
  - **OpenCV** - Video inpainting

## 🧪 API Testing

### Test Point Tracking

```bash
curl -X POST http://localhost:8000/api/track-point \
  -F "video=@test.mp4" \
  -F "x=50" \
  -F "y=50" \
  -F "frame_idx=0"
```

### Test Text-to-Video

```bash
curl -X POST http://localhost:8000/api/text-to-video \
  -F "video=@test.mp4" \
  -F "prompt=isolate drums"
```

### Test Object Removal

```bash
curl -X POST http://localhost:8000/api/remove-object \
  -F "video=@test.mp4" \
  -F "x=30" \
  -F "y=40" \
  -F "frame_idx=0"
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

## 📁 Project Structure

```
vtrack-ai-studio/
├── backend/
│   ├── server.py              # FastAPI app (SAM3 version)
│   ├── sam3_integration.py    # SAM3 unified API
│   ├── config.py              # Backend config
│   ├── requirements.txt       # Python deps
│   ├── SAM3_SETUP.md          # Setup guide
│   ├── checkpoints/           # Model checkpoints
│   │   └── sam3/              # SAM3 checkpoints
│   ├── sam3/                  # SAM3 repository
│   └── uploads/               # Temp uploads
├── src/
│   ├── lib/
│   │   └── api.ts             # API client
│   ├── pages/
│   │   └── Index.tsx          # Main page
│   ├── components/
│   │   ├── VideoUploader.tsx  # Video upload
│   │   ├── TabPanel.tsx       # 3 tabs UI
│   │   ├── DownloadPanel.tsx  # File downloads
│   │   └── ProcessingStatus.tsx # Progress
│   └── ...
├── MIGRATION.md               # SAM2 → SAM3 migration guide
├── package.json
├── .env                       # API URL config
└── README.md
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

### Backend Config

Edit `backend/config.py`:

```python
# SAM3 settings
SAM3_CHECKPOINT = CHECKPOINTS_DIR / "sam3" / "sam3_hiera_large.pt"
SAM3_CONFIG = "sam3_hiera_l.yaml"

# Video constraints
MAX_FILE_SIZE_MB = 100
MAX_VIDEO_DURATION = 10  # seconds
MAX_RESOLUTION = (854, 480)  # 480p
```

## 🐛 Troubleshooting

### Backend won't start

- Check SAM3 is installed: `cd backend/sam3 && pip install -e .`
- Download checkpoint: See setup instructions above
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Frontend can't connect to backend

- Check backend is running on port 8000
- Verify `.env` has correct `VITE_API_URL`
- Check CORS settings in `backend/config.py`

### Processing fails

- Check video constraints: ≤10s, ≤480p, <100MB
- Monitor backend logs for errors
- Ensure GPU has enough VRAM (16GB+)

### "SAM3 not available"

```bash
cd backend
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### "Checkpoint not found"

```bash
mkdir -p backend/checkpoints/sam3
wget -O backend/checkpoints/sam3/sam3_hiera_large.pt \
  https://dl.fbaipublicfiles.com/sam3/sam3_hiera_large.pt
```

## 📚 Documentation

- [SAM3 Setup Guide](backend/SAM3_SETUP.md) - Detailed setup instructions
- [Migration Guide](MIGRATION.md) - Migrate from SAM2 to SAM3
- [spec.md](spec.md) - Technical specification

## 📄 License

MIT

## 🔗 Related Projects

- [SAM 3](https://github.com/facebookresearch/sam3) - Meta's open-vocabulary segmentation model
- [Demucs](https://github.com/facebookresearch/demucs) - Audio source separation
- [VTrackAI](https://github.com/Amitro123/VTrackAI) - Original VTrackAI core modules

## 🙏 Acknowledgments

- **Meta AI** for SAM 3
- **Facebook Research** for Demucs
- **Lovable.dev** for the beautiful UI framework

---

**Built with ❤️ using SAM3, React, and FastAPI**
