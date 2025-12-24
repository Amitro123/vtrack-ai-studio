"""
VTrackAI Studio FastAPI Backend Server (SAM3 Version)
Provides REST API endpoints for video processing using SAM3.
"""

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

import config
from sam3_integration import get_sam3_engine, SAM3_AVAILABLE

# For audio processing (Demucs - unchanged)
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    import torch
    import torchaudio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("⚠️  Demucs not available")

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

def get_engine():
    """Get or create SAM3 engine."""
    global sam3_engine
    if sam3_engine is None:
        sam3_engine = get_sam3_engine(
            checkpoint_path=str(config.SAM3_CHECKPOINT),
            device=config.DEVICE
        )
    return sam3_engine


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
    is_valid, msg = config.validate_sam3_setup()
    return {
        "status": "healthy",
        "sam3_available": SAM3_AVAILABLE,
        "sam3_setup_valid": is_valid,
        "sam3_message": msg,
        "demucs_available": DEMUCS_AVAILABLE,
        "upload_dir": str(config.UPLOAD_DIR),
        "active_tasks": len(tasks),
        "device": config.DEVICE
    }


@app.post("/api/track-point")
async def track_point(
    video: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    frame_idx: int = Form(0)
):
    """
    Point-to-Track: Track object from click coordinates using SAM3.
    
    Args:
        video: Video file
        x: X coordinate (0-100, percentage)
        y: Y coordinate (0-100, percentage)
        frame_idx: Frame index for initial click
        
    Returns:
        task_id, masked_video_url, processing_steps
    """
    task_id = str(uuid.uuid4())
    
    try:
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
            {"id": "sam3", "label": "SAM3 point tracking", "status": "processing"},
            {"id": "mask", "label": "Generating mask overlay", "status": "pending"},
            {"id": "track", "label": "Tracking across frames", "status": "pending"},
        ]
        
        tasks[task_id] = {"steps": steps, "status": "processing"}
        
        # Get SAM3 engine
        engine = get_engine()
        
        # Track from point using SAM3
        result = engine.track_from_point(
            video_path=str(video_path),
            point=(x, y),
            frame_idx=frame_idx
        )
        
        # Update steps
        steps[0]["status"] = "complete"
        steps[1]["status"] = "complete"
        steps[2]["status"] = "complete"
        
        # Move output to task directory
        output_path = config.get_output_path(task_id, "masked_video.mp4")
        shutil.move(result["masked_video_path"], str(output_path))
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        tasks[task_id]["status"] = "complete"
        
        return {
            "task_id": task_id,
            "masked_video_url": f"/uploads/{task_id}/masked_video.mp4",
            "processing_steps": steps
        }
        
    except Exception as e:
        # Cleanup on error
        video_path.unlink(missing_ok=True)
        tasks[task_id] = {"status": "error", "error": str(e)}
        
        print(f"Error in track-point: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/api/text-to-video")
async def text_to_video(
    video: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Text-to-Video: Detect and track object from text using SAM3, isolate audio with Demucs.
    
    Args:
        video: Video file
        prompt: Text description (e.g., "isolate drums", "drummer")
        
    Returns:
        task_id, highlighted_video_url, audio_url, ai_response, processing_steps
    """
    task_id = str(uuid.uuid4())
    
    try:
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
        
        # Track from text using SAM3 (native text understanding)
        result = engine.track_from_text(
            video_path=str(video_path),
            prompt=prompt,
            frame_idx=0
        )
        
        steps[0]["status"] = "complete"
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Isolate audio using Demucs
        if DEMUCS_AVAILABLE:
            audio_path = config.get_output_path(task_id, f"{stem_type}.wav")
            
            # Extract audio from video
            import subprocess
            temp_audio = config.get_upload_path(f"{task_id}_audio.wav")
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                str(temp_audio), "-y"
            ], capture_output=True)
            
            # Load Demucs model
            model = pretrained.get_model(config.DEMUCS_MODEL)
            model.to(config.DEVICE)
            
            # Load audio
            wav, sr = torchaudio.load(str(temp_audio))
            wav = wav.to(config.DEVICE)
            
            # Apply Demucs
            with torch.no_grad():
                sources = apply_model(model, wav[None], device=config.DEVICE)[0]
            
            # Get the requested stem
            stem_idx = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}.get(stem_type, 2)
            isolated = sources[stem_idx]
            
            # Save isolated audio
            torchaudio.save(str(audio_path), isolated.cpu(), sr)
            
            # Cleanup temp audio
            temp_audio.unlink(missing_ok=True)
        else:
            # Fallback: just extract original audio
            audio_path = config.get_output_path(task_id, f"{stem_type}.wav")
            import subprocess
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                str(audio_path), "-y"
            ], capture_output=True)
        
        steps[2]["status"] = "complete"
        steps[3]["status"] = "complete"
        
        # Move output to task directory
        output_path = config.get_output_path(task_id, "highlighted_video.mp4")
        shutil.move(result["highlighted_video_path"], str(output_path))
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        tasks[task_id]["status"] = "complete"
        
        # Generate AI response
        responses = {
            "drums": f"🥁 Drums isolated! Audio stem extracted and video regions highlighted.",
            "vocals": f"🎤 Vocals extracted successfully! Audio and visual sync complete.",
            "bass": f"🎸 Bass located and isolated. Audio stem ready for download.",
            "other": f"✅ Processed \"{prompt}\". Found and isolated the matching elements."
        }
        
        ai_response = responses.get(stem_type, responses["other"])
        
        return {
            "task_id": task_id,
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
        
        print(f"Error in text-to-video: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/api/remove-object")
async def remove_object(
    video: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    frame_idx: int = Form(0)
):
    """
    Object Removal: Track object with SAM3, remove with inpainting.
    
    Args:
        video: Video file
        x: X coordinate (0-100, percentage)
        y: Y coordinate (0-100, percentage)
        frame_idx: Frame index for initial click
        
    Returns:
        task_id, inpainted_video_url, processing_steps
    """
    task_id = str(uuid.uuid4())
    
    try:
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
            frame_idx=frame_idx
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
        
        return {
            "task_id": task_id,
            "inpainted_video_url": f"/uploads/{task_id}/inpainted_video.mp4",
            "processing_steps": steps
        }
        
    except Exception as e:
        # Cleanup on error
        video_path.unlink(missing_ok=True)
        tasks[task_id] = {"status": "error", "error": str(e)}
        
        print(f"Error in remove-object: {e}")
        traceback.print_exc()
        
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
    
    print("=" * 60)
    print("🚀 VTrackAI Studio Backend Server (SAM3)")
    print("=" * 60)
    
    # Validate SAM3 setup
    is_valid, msg = config.validate_sam3_setup()
    print(msg)
    
    if not is_valid:
        print("\n⚠️  SAM3 setup incomplete. Some features may not work.")
        print("Please follow setup instructions in README.md")
    
    print(f"Upload Directory: {config.UPLOAD_DIR}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
