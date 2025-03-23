from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import uuid
import logging
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from podcast_agent.generator import get_turns
from podcast_agent.audio import generate_audio
from podcast_agent.visualizer import create_dot_visualization_video

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Podcast Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
os.makedirs("out", exist_ok=True)
os.makedirs("video", exist_ok=True)

# Serve static files
app.mount("/media", StaticFiles(directory="out"), name="media")
app.mount("/videos", StaticFiles(directory="video"), name="videos")

# Store job status
jobs: Dict[str, Dict[str, Any]] = {}

class PodcastRequest(BaseModel):
    topic: str
    waveform_color: Optional[str] = Field(default="#00FF00", pattern="^#[0-9A-Fa-f]{6}$")

class PodcastResponse(BaseModel):
    job_id: str
    topic: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None

def sanitize_filename(name: str) -> str:
    """Create a safe filename from the topic."""
    # Replace any non-alphanumeric chars with underscores
    return re.sub(r'[^a-zA-Z0-9]', '_', name)[:50]

async def generate_podcast_task(job_id: str, topic: str, waveform_color: str):
    try:
        # Update job status
        jobs[job_id]["status"] = "generating_script"
        jobs[job_id]["progress"] = 0.1
        
        logger.info(f"Generating script for job {job_id}, topic: {topic}")
        
        # Generate conversation turns
        turns = get_turns(topic)
        jobs[job_id]["progress"] = 0.3
        
        # Generate filename
        filename_topic = sanitize_filename(topic)
        unique_id = job_id[:8]
        audio_filename = f"out/{filename_topic}_{unique_id}.wav"
        
        # Update job status
        jobs[job_id]["status"] = "generating_audio"
        jobs[job_id]["progress"] = 0.4
        
        logger.info(f"Generating audio for job {job_id}")
        
        # Generate audio
        audio_file = generate_audio(turns, audio_filename)
        if not audio_file:
            raise Exception("Audio generation failed")
        
        jobs[job_id]["audio_url"] = f"/media/{os.path.basename(audio_file)}"
        jobs[job_id]["progress"] = 0.7
        
        # Update job status
        jobs[job_id]["status"] = "generating_video"
        
        logger.info(f"Generating video for job {job_id}")
        
        # Generate video visualization
        video_filename = f"video/{filename_topic}_{unique_id}_waveform.mp4"
        create_dot_visualization_video(audio_file, video_filename, color=waveform_color)
        
        jobs[job_id]["video_url"] = f"/videos/{os.path.basename(video_filename)}"
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating podcast (job {job_id}): {str(e)}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/api/podcast", response_model=PodcastResponse)
async def create_podcast(podcast_req: PodcastRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    # Validate color format
    try:
        if podcast_req.waveform_color and not re.match(r"^#[0-9A-Fa-f]{6}$", podcast_req.waveform_color):
            raise HTTPException(status_code=400, detail="Invalid color format. Use hex format like #00FF00")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid color format. Use hex format like #00FF00")
    
    # Initialize job status
    jobs[job_id] = {
        "topic": podcast_req.topic,
        "status": "queued",
        "progress": 0.0,
        "audio_url": None,
        "video_url": None,
        "waveform_color": podcast_req.waveform_color,
        "error": None
    }
    
    logger.info(f"Creating new podcast job {job_id} for topic: {podcast_req.topic}")
    
    # Start generation in background
    background_tasks.add_task(
        generate_podcast_task, 
        job_id=job_id, 
        topic=podcast_req.topic,
        waveform_color=podcast_req.waveform_color
    )
    
    return PodcastResponse(
        job_id=job_id,
        topic=podcast_req.topic,
        status="queued"
    )

@app.get("/api/podcast/{job_id}", response_model=JobStatusResponse)
async def get_podcast_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        audio_url=job["audio_url"],
        video_url=job["video_url"],
        error=job["error"]
    )

@app.get("/api/podcast")
async def list_podcasts():
    return JSONResponse(content=jobs)

@app.delete("/api/podcast/{job_id}")
async def delete_podcast(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete audio file if exists
    if job["audio_url"]:
        audio_path = os.path.join("out", os.path.basename(job["audio_url"].replace("/media/", "")))
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Deleted audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Error deleting audio file {audio_path}: {str(e)}")
    
    # Delete video file if exists
    if job["video_url"]:
        video_path = os.path.join("video", os.path.basename(job["video_url"].replace("/videos/", "")))
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Deleted video file: {video_path}")
            except Exception as e:
                logger.error(f"Error deleting video file {video_path}: {str(e)}")
    
    # Remove job from jobs dict
    del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
