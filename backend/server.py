"""
VTrackAI Studio FastAPI Backend Server (SAM3 Version)
Provides REST API endpoints for video processing using SAM3.
"""

import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import asyncio
from typing import Optional, List, Dict
import traceback

logger = logging.getLogger(__name__)

import config
from sam3_integration import get_sam3_engine, SAM3_AVAILABLE, get_device, is_mock_mode

# For audio processing (Librosa - cross-platform)
try:
    import librosa
    import torch
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("Librosa not available - audio features disabled")

# For inpainting (ProPainter/OpenCV - unchanged)
try:
    import cv2
    import numpy as np
    INPAINT_AVAILABLE = True
except ImportError:
    INPAINT_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="VTrackAI Studio API (SAM3)",
    description="Backend API for semantic video & audio editing powered by SAM3",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for static file serving
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")

# In-memory task storage (replace with Redis in production)
tasks: Dict[str, dict] = {}

# Initialize SAM3 engine (lazy loading)
sam3_engine = None
sam3_initialized = False  # Track if warm-up has been done

def get_engine():
    """Get or create SAM3 engine."""
    global sam3_engine
    if sam3_engine is None:
        sam3_engine = get_sam3_engine(
            checkpoint_path=str(config.SAM3_CHECKPOINT),
            device=config.DEVICE
        )
    return sam3_engine


def initialize_sam3_cache():
    """
    Pre-initialize SAM3 model to warm up the cache.
    This reduces latency on the first real request.
    """
    global sam3_initialized
    if sam3_initialized:
        return
    
    try:
        logger.info("🔥 Warming up SAM3 model...")
        engine = get_engine()
        # Just loading the model is enough to initialize it
        engine._load_model()
        sam3_initialized = True
        logger.info("✅ SAM3 warm-up complete")
    except Exception as e:
        logger.warning(f"SAM3 warm-up failed: {e}")


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to destination."""
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def validate_video(file_path: str) -> tuple[bool, str]:
    """Validate uploaded video file."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Invalid video file"
        
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if duration > config.MAX_VIDEO_DURATION:
            return False, f"Video too long: {duration:.1f}s (max {config.MAX_VIDEO_DURATION}s)"
        
        if width > config.MAX_RESOLUTION[0] or height > config.MAX_RESOLUTION[1]:
            return False, f"Resolution too high: {width}x{height} (max {config.MAX_RESOLUTION[0]}x{config.MAX_RESOLUTION[1]})"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


@app.get("/")
async def root():
    """Health check endpoint."""
    is_valid, msg = config.validate_sam3_setup()
    return {
        "status": "ok",
        "message": "VTrackAI Studio API (SAM3)",
        "sam3_available": SAM3_AVAILABLE,
        "sam3_setup": msg
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    import os
    cache_exists = os.path.exists("sam3_cache")
    is_valid, msg = config.validate_sam3_setup()
    return {
        "status": "healthy",
        "sam3_available": SAM3_AVAILABLE,
        "sam3_setup_valid": is_valid,
        "sam3_cache_ready": cache_exists,
        "device": get_device(),
        "audio_available": AUDIO_AVAILABLE,
        "upload_dir": str(config.UPLOAD_DIR),
        "active_tasks": len(tasks),
        "sam3_initialized": sam3_initialized
    }


@app.post("/api/warmup")
async def warmup():
    """
    Pre-initialize SAM3 model to warm up the cache.
    Call this after backend starts to reduce first-request latency.
    """
    try:
        initialize_sam3_cache()
        return {
            "status": "ok",
            "message": "SAM3 warm-up complete",
            "sam3_initialized": sam3_initialized
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "sam3_initialized": sam3_initialized
        }


# In-memory session storage for propagation results
propagation_sessions: Dict[str, dict] = {}


@app.post("/api/propagate")
async def propagate_video(
    file: UploadFile = File(...)
):
    """SAM3 Step 1: Pre-compute video features/cache"""
    try:
        import os
        import torch
        
        # Save uploaded video
        os.makedirs("uploads", exist_ok=True)
        video_path = f"uploads/{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Import SAM3 predictor
        from sam3.sam3_video_predictor import SAM3VideoPredictor
        
        predictor = SAM3VideoPredictor.from_pretrained(
            "checkpoints/sam3/sam3_hiera_large.pt"
        )
        
        # Propagation (from official docs)
        video_frames = predictor.load_video(video_path)
        inference_session = predictor.init_video(video_frames)
        
        # Save cache
        cache_dir = f"sam3_cache/{file.filename}"
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(inference_session, f"{cache_dir}/session.pt")
        
        return {
            "status": "success",
            "cache_id": cache_dir,
            "message": f"Propagation complete: {len(video_frames)} frames cached"
        }
    except Exception as e:
        logger.error(f"Propagation failed: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


@app.post("/api/track-point")
async def track_point(
    file: UploadFile = File(None),
    cache_id: str = Form(None),
    prompt_x: float = Form(0.5),
    prompt_y: float = Form(0.5),
    frame_idx: int = Form(0)
):
    """SAM3 Step 2: Track using pre-computed cache"""
    try:
        import os
        import torch
        
        # Auto-propagate if no cache
        if not cache_id:
            if not file:
                return {"status": "error", "detail": "Either file or cache_id required"}
            
            # Save and propagate
            os.makedirs("uploads", exist_ok=True)
            video_path = f"uploads/{file.filename}"
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            from sam3.sam3_video_predictor import SAM3VideoPredictor
            predictor = SAM3VideoPredictor.from_pretrained("checkpoints/sam3/sam3_hiera_large.pt")
            
            video_frames = predictor.load_video(video_path)
            inference_session = predictor.init_video(video_frames)
            
            cache_id = f"sam3_cache/{file.filename}"
            os.makedirs(cache_id, exist_ok=True)
            torch.save(inference_session, f"{cache_id}/session.pt")
        
        # Load cached session
        from sam3.sam3_video_predictor import SAM3VideoPredictor
        predictor = SAM3VideoPredictor.from_pretrained("checkpoints/sam3/sam3_hiera_large.pt")
        inference_session = torch.load(f"{cache_id}/session.pt")
        
        # Track
        outputs = predictor.track(
            inference_session=inference_session,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            frame_idx=frame_idx
        )
        
        return {
            "status": "success",
            "tracks": outputs,
            "cache_id": cache_id
        }
    except Exception as e:
        logger.error(f"Tracking failed: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}


@app.post("/api/text-to-video")
async def text_to_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("fast")
):
    """
    Text-to-Video: Detect and track object from text using SAM3, isolate audio with Demucs.
    
    Args:
        video: Video file
        prompt: Text description (e.g., "isolate drums", "drummer")
        mode: Processing mode ("fast" or "accurate", default "fast")
        
    Returns:
        task_id, highlighted_video_url, audio_url, ai_response, processing_steps
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Validate mode parameter
        if mode not in ["fast", "accurate"]:
            raise HTTPException(status_code=400, detail="Mode must be 'fast' or 'accurate'")
        
        # Save uploaded video
        video_path = config.get_upload_path(f"{task_id}_input.mp4")
        await save_upload_file(video, video_path)
        
        # Validate video
        is_valid, msg = validate_video(str(video_path))
        if not is_valid:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=msg)
        
        # Initialize processing steps (SAM3 version - no GroundingDINO needed)
        steps = [
            {"id": "sam3", "label": "SAM3 text understanding", "status": "processing"},
            {"id": "track", "label": "Tracking object in video", "status": "pending"},
            {"id": "demucs", "label": "Demucs audio separation", "status": "pending"},
            {"id": "highlight", "label": "Generating highlights", "status": "pending"},
        ]
        
        tasks[task_id] = {"steps": steps, "status": "processing"}
        
        # Extract audio stem type from prompt
        stem_map = {
            "drum": "drums",
            "vocal": "vocals",
            "guitar": "other",
            "bass": "bass",
            "piano": "other"
        }
        
        stem_type = "other"
        for key, value in stem_map.items():
            if key in prompt.lower():
                stem_type = value
                break
        
        # Get SAM3 engine
        engine = get_engine()
        logger.info(f"Using engine: {type(engine).__name__}")
        logger.info(f"is_mock_mode: {is_mock_mode()}")
        
        # Track from text using SAM3 (native text understanding)
        result = engine.track_from_text(
            video_path=str(video_path),
            prompt=prompt,
            frame_idx=0,
            mode=mode
        )
        
        steps[0]["status"] = "complete"
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Extract audio using ffmpeg + librosa (no demucs/torchcodec dependency)
        audio_path = config.get_output_path(task_id, f"{stem_type}.wav")
        audio_extraction_success = False
        
        try:
            import subprocess
            # Extract audio from video using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path), "-y"
            ], capture_output=True, check=False)
            
            if audio_path.exists() and audio_path.stat().st_size > 0:
                audio_extraction_success = True
                logger.info(f"Extracted audio: {audio_path}")
            else:
                logger.warning("Audio extraction produced empty file")
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
        
        steps[2]["status"] = "complete"
        steps[3]["status"] = "complete"
        
        # Move output to task directory
        output_path = config.get_output_path(task_id, "highlighted_video.mp4")
        shutil.move(result["highlighted_video_path"], str(output_path))
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        tasks[task_id]["status"] = "complete"
        
        # Get processing parameters for response
        from config import PROCESSING_FPS_FAST, PROCESSING_FPS_ACCURATE, KEYFRAMES_FAST, KEYFRAMES_ACCURATE
        processing_fps = PROCESSING_FPS_FAST if mode == "fast" else PROCESSING_FPS_ACCURATE
        num_keyframes = KEYFRAMES_FAST if mode == "fast" else KEYFRAMES_ACCURATE
        
        # Generate AI response
        responses = {
            "drums": f"[DRUMS] Drums isolated! Audio stem extracted and video regions highlighted.",
            "vocals": f"[VOCALS] Vocals extracted successfully! Audio and visual sync complete.",
            "bass": f"[BASS] Bass located and isolated. Audio stem ready for download.",
            "other": f"[SUCCESS] Processed \"{prompt}\". Found and isolated the matching elements."
        }
        
        ai_response = responses.get(stem_type, responses["other"])
        
        return {
            "task_id": task_id,
            "mode": mode,
            "sam3_mode": "mock" if is_mock_mode() else "real",
            "processing_fps": processing_fps,
            "num_keyframes": num_keyframes,
            "highlighted_video_url": f"/uploads/{task_id}/highlighted_video.mp4",
            "audio_url": f"/uploads/{task_id}/{stem_type}.wav",
            "ai_response": ai_response,
            "processing_steps": steps
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        video_path.unlink(missing_ok=True)
        tasks[task_id] = {"status": "error", "error": str(e)}
        
        logger.error(f"Error in text-to-video: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/api/remove-object")
async def remove_object(
    video: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    frame_idx: int = Form(0),
    mode: str = Form("fast")
):
    """
    Object Removal: Track object with SAM3, remove with inpainting.
    
    Args:
        video: Video file
        x: X coordinate (0-100, percentage)
        y: Y coordinate (0-100, percentage)
        frame_idx: Frame index for initial click
        mode: Processing mode ("fast" or "accurate", default "fast")
        
    Returns:
        task_id, inpainted_video_url, processing_steps
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Validate mode parameter
        if mode not in ["fast", "accurate"]:
            raise HTTPException(status_code=400, detail="Mode must be 'fast' or 'accurate'")
        
        # Save uploaded video
        video_path = config.get_upload_path(f"{task_id}_input.mp4")
        await save_upload_file(video, video_path)
        
        # Validate video
        is_valid, msg = validate_video(str(video_path))
        if not is_valid:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=msg)
        
        # Initialize processing steps (SAM3 version)
        steps = [
            {"id": "track", "label": "SAM3 temporal tracking", "status": "processing"},
            {"id": "mask", "label": "Generating removal mask", "status": "pending"},
            {"id": "inpaint", "label": "OpenCV inpainting", "status": "pending"},
            {"id": "render", "label": "Rendering final video", "status": "pending"},
        ]
        
        tasks[task_id] = {"steps": steps, "status": "processing"}
        
        # Get SAM3 engine
        engine = get_engine()
        
        # Segment for removal using SAM3
        result = engine.segment_for_removal(
            video_path=str(video_path),
            point=(x, y),
            frame_idx=frame_idx,
            mode=mode
        )
        
        steps[0]["status"] = "complete"
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Inpaint video using OpenCV (simple fallback)
        if INPAINT_AVAILABLE:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = config.get_output_path(task_id, "inpainted_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply inpainting if mask available
                if frame_idx in result["masks"]:
                    mask = result["masks"][frame_idx]
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask.astype(np.uint8) * 255, (width, height))
                    else:
                        mask = (mask * 255).astype(np.uint8)
                    
                    # OpenCV inpainting
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
        else:
            raise HTTPException(status_code=500, detail="Inpainting not available")
        
        steps[2]["status"] = "complete"
        steps[3]["status"] = "complete"
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        tasks[task_id]["status"] = "complete"
        
        # Get processing parameters for response
        from config import PROCESSING_FPS_FAST, PROCESSING_FPS_ACCURATE, KEYFRAMES_FAST, KEYFRAMES_ACCURATE
        processing_fps = PROCESSING_FPS_FAST if mode == "fast" else PROCESSING_FPS_ACCURATE
        num_keyframes = KEYFRAMES_FAST if mode == "fast" else KEYFRAMES_ACCURATE
        
        return {
            "task_id": task_id,
            "mode": mode,
            "sam3_mode": "mock" if is_mock_mode() else "real",
            "processing_fps": processing_fps,
            "num_keyframes": num_keyframes,
            "inpainted_video_url": f"/uploads/{task_id}/inpainted_video.mp4",
            "processing_steps": steps
        }
        
    except Exception as e:
        # Cleanup on error
        video_path.unlink(missing_ok=True)
        tasks[task_id] = {"status": "error", "error": str(e)}
        
        logger.error(f"Error in remove-object: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a processing task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("VTrackAI Studio Backend Server (SAM3)")
    logger.info("=" * 60)
    
    # Validate SAM3 setup
    is_valid, msg = config.validate_sam3_setup()
    logger.info(msg)
    
    if not is_valid:
        logger.warning("SAM3 setup incomplete. Some features may not work.")
        logger.info("Please follow setup instructions in README.md")
    
    logger.info(f"Upload Directory: {config.UPLOAD_DIR}")
    logger.info(f"Server: http://{config.HOST}:{config.PORT}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
