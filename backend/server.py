"""
VTrackAI Studio FastAPI Backend Server
Provides REST API endpoints for video processing using VTrackAI core modules.
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
from core_integration import vtrack_core, VTRACK_AVAILABLE, video_utils, audio_utils

# Initialize FastAPI app
app = FastAPI(
    title="VTrackAI Studio API",
    description="Backend API for semantic video & audio editing",
    version="1.0.0"
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


def validate_video(file_path: str) -> tuple[bool, str]:
    """Validate uploaded video file."""
    try:
        is_valid, msg = video_utils.validate_video(file_path)
        return is_valid, msg
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to destination."""
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "VTrackAI Studio API",
        "vtrack_available": VTRACK_AVAILABLE
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "vtrack_core_available": VTRACK_AVAILABLE,
        "upload_dir": str(config.UPLOAD_DIR),
        "active_tasks": len(tasks)
    }


@app.post("/api/track-point")
async def track_point(
    video: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    frame_idx: int = Form(0)
):
    """
    Point-to-Track: Track object from click coordinates using SAM2.
    
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
        
        # Initialize processing steps
        steps = [
            {"id": "sam2", "label": "SAM2 point tracking", "status": "processing"},
            {"id": "mask", "label": "Generating mask overlay", "status": "pending"},
            {"id": "track", "label": "Tracking across frames", "status": "pending"},
        ]
        
        tasks[task_id] = {"steps": steps, "status": "processing"}
        
        # Get SAM2 tracker
        tracker = vtrack_core.get_sam2_tracker()
        
        # Track from point
        masks = tracker.track_from_point(
            str(video_path),
            point=(x, y),
            frame_idx=frame_idx
        )
        
        # Update steps
        steps[0]["status"] = "complete"
        steps[1]["status"] = "processing"
        
        # Create masked video
        frames = video_utils.extract_frames(str(video_path))
        masked_frames = []
        
        for idx, frame in enumerate(frames):
            if idx in masks:
                overlay = tracker.get_mask_overlay(frame, masks[idx], color=(255, 0, 0))
                masked_frames.append(overlay)
            else:
                masked_frames.append(frame)
        
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Save masked video
        video_info = video_utils.get_video_info(str(video_path))
        output_path = config.get_output_path(task_id, "masked_video.mp4")
        video_utils.create_video_from_frames(
            masked_frames,
            str(output_path),
            fps=video_info['fps']
        )
        
        steps[2]["status"] = "complete"
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        # Clear GPU cache
        vtrack_core.clear_cache()
        
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
    Text-to-Video: Detect object from text, track with SAM2, isolate audio with Demucs.
    
    Args:
        video: Video file
        prompt: Text description (e.g., "isolate drums")
        
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
        
        # Initialize processing steps
        steps = [
            {"id": "grounding", "label": "GroundingDINO text→bbox", "status": "processing"},
            {"id": "sam2", "label": "SAM2 segmentation", "status": "pending"},
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
        
        # Get Grounding DINO
        dino = vtrack_core.get_grounding_dino()
        
        # Detect object from text
        initial_mask = dino.get_initial_mask(str(video_path), prompt, frame_idx=0)
        
        steps[0]["status"] = "complete"
        steps[1]["status"] = "processing"
        
        # Get SAM2 tracker
        tracker = vtrack_core.get_sam2_tracker()
        
        # Track from initial mask
        if initial_mask is not None:
            masks = tracker.track_from_mask(str(video_path), initial_mask, frame_idx=0)
        else:
            # Fallback: no detection, return error
            raise HTTPException(
                status_code=404,
                detail=f"Could not detect '{prompt}' in video"
            )
        
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Get SAM Audio
        sam_audio = vtrack_core.get_sam_audio()
        
        # Isolate audio stem
        audio_path = config.get_output_path(task_id, f"{stem_type}.wav")
        sam_audio.process_video(
            str(video_path),
            stem_type,
            str(audio_path),
            masks
        )
        
        steps[2]["status"] = "complete"
        steps[3]["status"] = "processing"
        
        # Create highlighted video
        frames = video_utils.extract_frames(str(video_path))
        highlighted_frames = []
        
        for idx, frame in enumerate(frames):
            if idx in masks:
                overlay = tracker.get_mask_overlay(frame, masks[idx], color=(0, 255, 0), alpha=0.3)
                highlighted_frames.append(overlay)
            else:
                highlighted_frames.append(frame)
        
        video_info = video_utils.get_video_info(str(video_path))
        output_video_path = config.get_output_path(task_id, "highlighted_video.mp4")
        video_utils.create_video_from_frames(
            highlighted_frames,
            str(output_video_path),
            fps=video_info['fps']
        )
        
        steps[3]["status"] = "complete"
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        # Clear GPU cache
        vtrack_core.clear_cache()
        
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
    Object Removal: Track object with SAM2, remove with ProPainter inpainting.
    
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
        
        # Initialize processing steps
        steps = [
            {"id": "track", "label": "SAM2 temporal tracking", "status": "processing"},
            {"id": "mask", "label": "Generating removal mask", "status": "pending"},
            {"id": "inpaint", "label": "ProPainter inpainting", "status": "pending"},
            {"id": "render", "label": "Rendering final video", "status": "pending"},
        ]
        
        tasks[task_id] = {"steps": steps, "status": "processing"}
        
        # Get SAM2 tracker
        tracker = vtrack_core.get_sam2_tracker()
        
        # Track from point
        masks = tracker.track_from_point(
            str(video_path),
            point=(x, y),
            frame_idx=frame_idx
        )
        
        steps[0]["status"] = "complete"
        steps[1]["status"] = "processing"
        
        # Prepare masks for inpainting
        steps[1]["status"] = "complete"
        steps[2]["status"] = "processing"
        
        # Get ProPainter
        propainter = vtrack_core.get_propainter()
        
        # Inpaint video
        output_path = config.get_output_path(task_id, "inpainted_video.mp4")
        propainter.inpaint_video(
            str(video_path),
            masks,
            str(output_path)
        )
        
        steps[2]["status"] = "complete"
        steps[3]["status"] = "processing"
        
        # Rendering complete
        steps[3]["status"] = "complete"
        
        # Cleanup input video
        video_path.unlink(missing_ok=True)
        
        # Clear GPU cache
        vtrack_core.clear_cache()
        
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
    print("🚀 VTrackAI Studio Backend Server")
    print("=" * 60)
    print(f"VTrackAI Core Available: {VTRACK_AVAILABLE}")
    print(f"Upload Directory: {config.UPLOAD_DIR}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
