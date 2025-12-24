# VTrackAI Studio

**Semantic Video & Audio Editor** - Professional UI for VTrackAI powered by React + FastAPI

[![React](https://img.shields.io/badge/React-18.3-blue?logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?logo=typescript)](https://www.typescriptlang.org/)

## 🌟 Features

- **Point-to-Track**: Click any object → SAM2 tracks across frames
- **Text-to-Video**: "Isolate drums" → GroundingDINO + Demucs extracts audio
- **Object Removal**: Click to remove → ProPainter inpaints seamlessly
- **Single Upload**: One video upload shared across all tabs
- **Real-time Progress**: Live processing status from backend
- **Download Results**: Processed videos and isolated audio files

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **CUDA GPU** with 12GB+ VRAM (A10G/T4 or better)
- **VTrackAI** repository cloned in parent directory

### 1. Start Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
```

Backend runs on **http://localhost:8000**

### 2. Start Frontend

```bash
npm install
npm run dev
```

Frontend runs on **http://localhost:5173**

### 3. Open Browser

Navigate to **http://localhost:5173** and start editing!

## 📖 How to Use

### Step 1: Upload Video

- Click the upload area at the top
- Select a video (max 10s, 480p, <100MB)
- Supported formats: MP4, WebM, MOV, AVI, MKV
- Video is automatically shared across all tabs

### Step 2: Choose Your Workflow

#### Tab 1: Click to Track 🎯

1. Click on any object in the video
2. SAM2 generates a red mask overlay
3. Mask tracks across all frames
4. Download the masked video

**Use Case**: Track a person, object, or instrument through the video

#### Tab 2: Chat (Text-to-Video) 💬

1. Type a text prompt:
   - "Isolate drums"
   - "Extract vocals"
   - "Find the guitar"
2. GroundingDINO detects the object
3. SAM2 segments and tracks it
4. Demucs separates the audio
5. Download isolated audio + highlighted video

**Use Case**: Extract specific instrument audio from band performances

#### Tab 3: Remove Objects 🗑️

1. Click on an object to remove
2. SAM2 tracks it across frames
3. ProPainter inpaints to remove it
4. Download the cleaned video

**Use Case**: Remove unwanted objects or people from videos

## 🎬 Example Workflow: Band Demo

1. **Upload** a 10s band performance video
2. **Tab 2 (Chat)**: Type "Isolate drums"
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
- **VTrackAI Core** (AI models)
  - SAM 2 (Meta) - Video object tracking
  - Demucs - Audio stem separation
  - GroundingDINO - Text-to-bounding-box
  - ProPainter - Video inpainting

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
│   ├── server.py              # FastAPI app
│   ├── core_integration.py    # VTrackAI imports
│   ├── config.py              # Backend config
│   ├── requirements.txt       # Python deps
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
MAX_FILE_SIZE_MB = 100
MAX_VIDEO_DURATION = 10  # seconds
MAX_RESOLUTION = (854, 480)  # 480p
```

## 🐛 Troubleshooting

### Backend won't start

- Check VTrackAI is in parent directory: `../VTrackAI`
- Install dependencies: `pip install -r backend/requirements.txt`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Frontend can't connect to backend

- Check backend is running on port 8000
- Verify `.env` has correct `VITE_API_URL`
- Check CORS settings in `backend/config.py`

### Processing fails

- Check video constraints: ≤10s, ≤480p, <100MB
- Monitor backend logs for errors
- Ensure GPU has enough VRAM (12GB+)

## 📄 License

MIT

## 🔗 Related Projects

- [VTrackAI](https://github.com/Amitro123/VTrackAI) - Core AI modules
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) - Meta's segmentation model
- [Demucs](https://github.com/facebookresearch/demucs) - Audio source separation
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Text-to-detection

---

**Built with ❤️ using Lovable.dev**
