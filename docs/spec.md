# VTrackAI Studio - Technical Specification

## Overview

VTrackAI Studio is a semantic video and audio editor that uses Meta's SAM 3 (Segment Anything Model 3) for open-vocabulary video segmentation. The application provides a professional React-based UI for three main workflows: point-to-track, text-to-video, and object removal. GPU-first architecture optimized for Colab deployment.

**Version**: 2.0.0 (SAM3 GPU-First)  
**Architecture**: React + TypeScript (Frontend) + FastAPI + SAM3 (Backend)  
**License**: MIT

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Browser)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              React Frontend (Vite + TypeScript)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ VideoUploader│  │   TabPanel   │  │DownloadPanel │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Client (api.ts)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           FastAPI Backend (Python 3.12)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   server.py                          │  │
│  │  /api/track-point  /api/text-to-video  /api/remove  │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            sam3_integration.py                       │  │
│  │         (SAM3VideoEngine)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│          ┌──────────────┼──────────────┐                   │
│          ▼              ▼              ▼                   │
│     ┌────────┐    ┌─────────┐    ┌────────┐              │
│     │  SAM3  │    │ Demucs  │    │ OpenCV │              │
│     │(Video) │    │ (Audio) │    │(Inpaint│              │
│     └────────┘    └─────────┘    └────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
Frontend Components:
├── App.tsx                    # Root component, routing
├── pages/
│   └── Index.tsx              # Main page, state management
├── components/
│   ├── Header.tsx             # App header
│   ├── VideoUploader.tsx      # Video upload + player
│   ├── TabPanel.tsx           # 3 tabs (Click, Chat, Remove)
│   ├── DownloadPanel.tsx      # Download results
│   └── ProcessingStatus.tsx   # Progress indicator
└── lib/
    └── api.ts                 # Backend API client

Backend Modules:
├── server.py                  # FastAPI app, endpoints
├── sam3_integration.py        # SAM3 wrapper
├── config.py                  # Configuration
└── requirements.txt           # Dependencies
```

---

## Core Features

### 1. Point-to-Track

**Purpose**: Track any object in a video by clicking on it

**Flow**:
1. User uploads video
2. User clicks on object in video player
3. Frontend sends point coordinates to `/api/track-point`
4. Backend uses SAM3 to segment and track object
5. Returns masked video with red overlay
6. User downloads result

**Technology**: SAM3 point prompts

**Processing Steps**:
- SAM3 point tracking
- Generating mask overlay
- Tracking across frames

### 2. Text-to-Video

**Purpose**: Detect and track objects using natural language, isolate audio

**Flow**:
1. User uploads video
2. User types text prompt (e.g., "isolate drums")
3. Frontend sends prompt to `/api/text-to-video`
4. Backend uses SAM3 native text understanding to detect object
5. SAM3 tracks object across frames
6. Demucs isolates audio stem
7. Returns highlighted video + isolated audio
8. User downloads both results

**Technology**: SAM3 text prompts + Demucs

**Processing Steps**:
- SAM3 text understanding
- Tracking object in video
- Demucs audio separation
- Generating highlights

### 3. Object Removal

**Purpose**: Remove unwanted objects from video

**Flow**:
1. User uploads video
2. User clicks on object to remove
3. Frontend sends coordinates to `/api/remove-object`
4. Backend uses SAM3 to track object
5. OpenCV inpaints to remove object
6. Returns inpainted video
7. User downloads result

**Technology**: SAM3 tracking + OpenCV inpainting

**Processing Steps**:
- SAM3 temporal tracking
- Generating removal mask
- OpenCV inpainting
- Rendering final video

---

## API Specification

### Base URL

```
http://localhost:8000
```

### Endpoints

#### GET /

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "message": "VTrackAI Studio API (SAM3)",
  "sam3_available": true,
  "sam3_setup": "✅ SAM3 setup valid (device: cuda, VRAM: 16GB)"
}
```

#### GET /api/health

Detailed health check.

**Response**:
```json
{
  "status": "healthy",
  "sam3_available": true,
  "sam3_setup_valid": true,
  "device": "cuda",
  "audio_available": true,
  "upload_dir": "/path/to/uploads",
  "active_tasks": 0
}
```

#### POST /api/track-point

Track object from point click.

**Request** (multipart/form-data):
- `video`: Video file (MP4, WebM, MOV, AVI, MKV)
- `x`: X coordinate (0-100, percentage)
- `y`: Y coordinate (0-100, percentage)
- `frame_idx`: Frame index (default: 0)
- `mode`: Processing mode - "fast" (default) or "accurate"

**Response**:
```json
{
  "task_id": "uuid",
  "mode": "fast",
  "processing_fps": 5.0,
  "num_keyframes": 12,
  "masked_video_url": "/uploads/{task_id}/masked_video.mp4",
  "processing_steps": [
    {"id": "sam3", "label": "SAM3 point tracking", "status": "complete"},
    {"id": "mask", "label": "Generating mask overlay", "status": "complete"},
    {"id": "track", "label": "Tracking across frames", "status": "complete"}
  ]
}
```

#### POST /api/text-to-video

Detect and track object from text, isolate audio.

**Request** (multipart/form-data):
- `video`: Video file
- `prompt`: Text description (e.g., "isolate drums", "drummer")
- `mode`: Processing mode - "fast" (default) or "accurate"

**Response**:
```json
{
  "task_id": "uuid",
  "mode": "fast",
  "processing_fps": 5.0,
  "num_keyframes": 12,
  "highlighted_video_url": "/uploads/{task_id}/highlighted_video.mp4",
  "audio_url": "/uploads/{task_id}/drums.wav",
  "ai_response": "🥁 Drums isolated! Audio stem extracted and video regions highlighted.",
  "processing_steps": [
    {"id": "sam3", "label": "SAM3 text understanding", "status": "complete"},
    {"id": "track", "label": "Tracking object in video", "status": "complete"},
    {"id": "demucs", "label": "Demucs audio separation", "status": "complete"},
    {"id": "highlight", "label": "Generating highlights", "status": "complete"}
  ]
}
```

#### POST /api/remove-object

Remove object from video.

**Request** (multipart/form-data):
- `video`: Video file
- `x`: X coordinate (0-100, percentage)
- `y`: Y coordinate (0-100, percentage)
- `frame_idx`: Frame index (default: 0)
- `mode`: Processing mode - "fast" (default) or "accurate"

**Response**:
```json
{
  "task_id": "uuid",
  "mode": "fast",
  "processing_fps": 5.0,
  "num_keyframes": 12,
  "inpainted_video_url": "/uploads/{task_id}/inpainted_video.mp4",
  "processing_steps": [
    {"id": "track", "label": "SAM3 temporal tracking", "status": "complete"},
    {"id": "mask", "label": "Generating removal mask", "status": "complete"},
    {"id": "inpaint", "label": "OpenCV inpainting", "status": "complete"},
    {"id": "render", "label": "Rendering final video", "status": "complete"}
  ]
}
```

#### GET /api/task/{task_id}

Get task status.

**Response**:
```json
{
  "steps": [...],
  "status": "complete" | "processing" | "error",
  "error": "error message (if failed)"
}
```

---

## Data Flow

### Point-to-Track Flow

```
User clicks (x, y) on video
    ↓
Frontend: api.trackPoint(video, x, y, frameIdx)
    ↓
Backend: POST /api/track-point
    ↓
SAM3VideoEngine.track_from_point(video_path, point)
    ↓
SAM3: Initialize inference state
    ↓
SAM3: Add point prompt on frame
    ↓
SAM3: Propagate masks through video
    ↓
Create masked video with overlay
    ↓
Return: {masked_video_url, masks, tracks}
    ↓
Frontend: Display result, enable download
```

### Text-to-Video Flow

```
User types "isolate drums"
    ↓
Frontend: api.textToVideo(video, prompt)
    ↓
Backend: POST /api/text-to-video
    ↓
SAM3VideoEngine.track_from_text(video_path, prompt)
    ↓
SAM3: Initialize inference state
    ↓
SAM3: Add text prompt (native understanding)
    ↓
SAM3: Propagate masks through video
    ↓
Create highlighted video
    ↓
Demucs: Isolate audio stem (drums)
    ↓
Return: {highlighted_video_url, audio_url, ai_response}
    ↓
Frontend: Display results, enable downloads
```

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.3 | UI framework |
| TypeScript | 5.8 | Type safety |
| Vite | 5.4 | Build tool |
| shadcn/ui | Latest | UI components |
| Tailwind CSS | 3.x | Styling |
| TanStack Query | Latest | Data fetching |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12 | Runtime |
| FastAPI | 0.104+ | REST API framework |
| Uvicorn | Latest | ASGI server |
| PyTorch | 2.7.0+ (2.9.1 recommended) | Deep learning |
| CUDA | 12.6+ | GPU acceleration |

### AI Models

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| SAM 3 | Video segmentation & tracking | Point/text prompt + video | Masks + tracks |
| Demucs | Audio stem separation | Audio waveform | Isolated stems |
| OpenCV | Video inpainting | Video + mask | Inpainted video |

### SAM3 Setup Requirements

**Hugging Face Authentication Required:**

SAM3 model checkpoints are hosted on Hugging Face and require authentication:

1. **Request Access**: Visit https://huggingface.co/facebook/sam3
2. **Create Token**: Generate a Hugging Face access token
3. **Authenticate**: Run `huggingface-cli login`
4. **Download**: Use `hf_hub_download()` to get checkpoints

**Dependencies:**
- `huggingface-hub>=0.20.0` (for checkpoint downloads)
- PyTorch 2.7.0+ with CUDA 12.6+
- SAM3 package installed from GitHub

**See**: [`backend/SAM3_AUTHENTICATION_GUIDE.md`](../backend/SAM3_AUTHENTICATION_GUIDE.md) for detailed setup instructions.

---

## Configuration

### Backend Configuration (`backend/config.py`)

```python
# SAM3 settings
SAM3_CHECKPOINT = "checkpoints/sam3/sam3_hiera_large.pt"
SAM3_CONFIG = "sam3_hiera_l.yaml"

# Hugging Face (for checkpoint downloads)
# Requires authentication: huggingface-cli login
HF_REPO_ID = "facebook/sam3"
HF_CHECKPOINT_FILENAME = "sam3_hiera_large.pt"

# Server settings
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", ...]

# Video constraints
MAX_FILE_SIZE_MB = 100
MAX_VIDEO_DURATION = 10  # seconds
MAX_RESOLUTION = (854, 480)  # 480p
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# GPU settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = True
MAX_VRAM_GB = 16

# Processing modes (optimization)
PROCESSING_FPS_FAST = 5.0       # Fast mode: 5 FPS processing
PROCESSING_FPS_ACCURATE = 10.0  # Accurate mode: 10 FPS processing
KEYFRAMES_FAST = 12             # Fast mode: 12 keyframes
KEYFRAMES_ACCURATE = 32         # Accurate mode: 32 keyframes
```

### Frontend Configuration (`.env`)

```env
VITE_API_URL=http://localhost:8000
```

---

## Performance Requirements

### Video Constraints

- **Max Duration**: 10 seconds
- **Max Resolution**: 480p (854x480)
- **Max File Size**: 100MB
- **Supported Formats**: MP4, WebM, MOV, AVI, MKV

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | A10G/A100 (24GB+) |
| CUDA | 12.6 | 12.6+ |
| RAM | 16GB | 32GB+ |
| Storage | 10GB | 50GB+ |

### Performance Benchmarks

| Workflow | SAM2 + GroundingDINO | SAM3 | Improvement |
|----------|---------------------|------|-------------|
| Point-to-Track | 8.5s | 6.2s | **27% faster** |
| Text-to-Video | 12.3s | 9.1s | **26% faster** |
| Object Removal | 15.7s | 13.4s | **15% faster** |
| VRAM Usage | 13.8GB | 11.2GB | **19% less** |

---

## Performance Optimization

### Processing Modes

VTrackAI Studio offers two processing modes to balance speed and quality:

#### Fast Mode (Default)
- **FPS**: 5.0 (downsampled from original)
- **Keyframes**: 12 frames processed by SAM3
- **Speed**: ~70% faster than dense processing
- **Use Case**: Quick previews, rapid iteration
- **Quality**: Good temporal accuracy for most scenarios

#### Accurate Mode
- **FPS**: 10.0 (downsampled from original)
- **Keyframes**: 32 frames processed by SAM3
- **Speed**: ~40-50% faster than dense processing
- **Use Case**: Final outputs, complex tracking
- **Quality**: Better temporal consistency, smoother transitions

### Optimization Techniques

1. **FPS Downsampling**
   - Original video downsampled to target FPS
   - Reduces number of frames to process
   - Frame mapping maintained for reconstruction

2. **Keyframe-Based Processing**
   - SAM3 runs only on selected keyframes
   - Uniform sampling ensures temporal coverage
   - First and last frames always included

3. **Mask Interpolation**
   - Non-keyframe masks interpolated from nearest keyframe
   - Simple nearest-neighbor strategy
   - Can be enhanced with optical flow in future

### Performance Comparison

| Metric | Dense Processing | Fast Mode | Accurate Mode |
|--------|-----------------|-----------|---------------|
| FPS Processed | 25 (original) | 5 | 10 |
| Keyframes | ~50 | 12 | 32 |
| Processing Time | 100% | ~30% | ~50-60% |
| Quality | Baseline | Good | Excellent |
| VRAM Usage | High | Medium | Medium-High |

### Module: video_preprocessor.py

Core optimization functions:

```python
def downsample_video(video_path, target_fps) -> (path, mapping):
    """Downsample video to target FPS"""
    
def select_keyframes(num_frames, k) -> List[int]:
    """Select k keyframes uniformly"""
    
def interpolate_masks(keyframe_masks, keyframe_indices, total_frames) -> Dict:
    """Interpolate masks for non-keyframe frames"""
```

---

## Security Considerations

### File Upload

- File size validation (max 100MB)
- File type validation (whitelist)
- Video duration validation (max 10s)
- Resolution validation (max 480p)

### CORS

- Configured origins in `config.py`
- Credentials allowed for local development
- Should be restricted in production

### Data Privacy

- Uploaded files stored temporarily
- Auto-cleanup after 24 hours
- No data persistence beyond session

---

## Error Handling

### Backend Errors

| Error | HTTP Code | Description |
|-------|-----------|-------------|
| Invalid video | 400 | File too large, wrong format, etc. |
| SAM3 not available | 500 | SAM3 not installed |
| Checkpoint not found | 500 | Model checkpoint missing |
| CUDA out of memory | 500 | Insufficient VRAM |
| Processing failed | 500 | Generic processing error |

### Frontend Error Handling

- Toast notifications for errors
- Graceful degradation
- Retry mechanisms
- User-friendly error messages

---

## Deployment

### Development

```bash
# Backend
conda activate sam3
cd backend
python server.py

# Frontend
npm run dev
```

### Production

```bash
# Backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app

# Frontend
npm run build
# Serve dist/ with nginx or similar
```

### Google Colab (GPU Backend)

One-click GPU deployment with ngrok:

1. Open `colab/vtrackai_sam3_gpu.ipynb` in Colab
2. Run all cells
3. Copy the ngrok URL
4. Set `VITE_API_URL=<ngrok-url>` in local frontend

See `colab/vtrackai_sam3_gpu.ipynb` for full setup.

---

## Testing

### Test Suite Overview

Comprehensive test coverage with ~37 tests:

#### Unit Tests (`test_video_preprocessor.py`)
- Keyframe selection logic
- Mask interpolation algorithms
- Processing parameter retrieval
- Edge cases (single frame, zero frames, etc.)

#### Integration Tests (`test_api_integration.py`)
- All API endpoints (health, track-point, text-to-video, remove-object)
- Fast and Accurate mode validation
- Mode parameter validation
- Response format verification
- Error handling (invalid modes, missing parameters)

#### E2E Tests (`test_e2e_pipeline.py`)
- Full preprocessing pipeline (Fast mode)
- Full preprocessing pipeline (Accurate mode)
- Video downsampling verification
- Mask interpolation quality
- Processing mode consistency
- Edge case handling

### Running Tests

```bash
# Install test dependencies
pip install -r backend/requirements-test.txt

# Run all tests
python -m pytest backend/ -v

# Run with coverage
python -m pytest backend/ -v --cov=backend --cov-report=html

# Run specific test file
python -m pytest backend/test_video_preprocessor.py -v
```

### Test Configuration

- **Pytest markers**: integration, e2e, slow
- **Mocked SAM3**: Integration tests use mocked engine
- **Temporary videos**: E2E tests create temp test videos
- **CI/CD ready**: Tests run without GPU requirements

See [TESTING.md](../backend/TESTING.md) for detailed documentation.

---

## Future Enhancements

### Short-term

- [ ] WebSocket support for real-time progress
- [ ] Batch processing
- [ ] Video preview in results
- [ ] More audio stems

### Long-term

- [ ] User authentication
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Real-time video processing
- [ ] Multi-user support

---

## References

- [SAM 3 Paper](https://github.com/facebookresearch/sam3)
- [Demucs](https://github.com/facebookresearch/demucs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

**Last Updated**: 2025-12-25  
**Version**: 2.0.0 (SAM3 + Optimizations)
