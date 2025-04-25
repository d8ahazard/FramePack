import asyncio
import json
import os
import time
import traceback
from typing import List, Union
from typing import Optional

from fastapi import BackgroundTasks, HTTPException
from starlette.responses import FileResponse

from datatypes.datatypes import GenerateResponse, JobStatus, SegmentConfig, VideoRequest, \
    DeleteJobRequest
from datatypes.datatypes import JobStatusResponse, ErrorResponse
from datatypes.datatypes import SaveJobRequest
from handlers.path import job_path
from handlers.path import upload_path

#from infer import manager

# Job queue and processing state
job_queue = []
running_job_id = None
job_processing_lock = asyncio.Lock()

# Global job tracking
job_statuses = {}


# Run this at app start in case we terminated/failed
def clear_running_jobs():
    # Get all saved jobs
    saved_jobs = list_saved_jobs()
    # Check if any jobs are running
    for job_id in saved_jobs:
        job_data = load_job_data(job_id)
        if job_data:
            # Check if the job is running
            if job_data.get("status") == "running":
                # If so, set it to failed
                job_data["status"] = "failed"
                job_data["message"] = "Job was interrupted"
                save_job_data(job_id, job_data)


def add_to_queue(job_id: str, position: Optional[int] = None):
    """
    Add a job to the queue for processing

    Args:
        job_id: The unique job identifier
        position: Optional position in queue (0-based, None = end of queue)
    """
    global job_queue

    # If position is specified and valid, insert at that position
    if position is not None and 0 <= position < len(job_queue):
        job_queue.insert(position, job_id)
    else:
        # Otherwise, add to the end
        job_queue.append(job_id)

    # Update all queue positions
    update_queue_positions()

    # Start queue processing if needed
    asyncio.create_task(process_queue())


def save_job_data(job_id, job_data):
    """Save job data to disk for persistence."""

    # For compatibility: convert different types of input to dict
    if isinstance(job_data, JobStatus):
        data_dict = job_data.to_dict()
    elif hasattr(job_data, "model_dump"):
        # Handle Pydantic models
        data_dict = job_data.model_dump()
    elif hasattr(job_data, "dict"):
        # Handle older Pydantic models
        data_dict = job_data.dict()
    else:
        # Assume it's already a dict
        data_dict = job_data

    # Ensure job_id is consistent
    data_dict["job_id"] = job_id

    # Add timestamp if missing
    if "created_timestamp" not in data_dict:
        data_dict["created_timestamp"] = int(time.time())

    # Ensure job_path directory exists
    os.makedirs(job_path, exist_ok=True)

    # Save to disk
    job_file = os.path.join(job_path, f"{job_id}.json")

    try:
        with open(job_file, "w") as f:
            json.dump(data_dict, f, indent=4)

    except Exception as e:
        print(f"Error saving job data: {e}")
        raise e

    # Broadcast update to WebSocket clients
    try:
        # Use the update_status function from socket.py to handle the websocket broadcasting
        from handlers.socket import update_status_sync
        # Extract key status fields if they exist
        status = data_dict.get("status")
        progress = data_dict.get("progress")
        message = data_dict.get("message")
        # Update with all data for complete information
        update_status_sync(job_id, status, progress, message, data_dict)
    except Exception as e:
        print(f"Error queuing job update for broadcast: {e}")
        traceback.print_exc()

    return True


def load_job_data(job_id):
    """
    Load job data from a JSON file

    Args:
        job_id: Unique job ID

    Returns:
        dict: Job data or None if not found
    """
    job_file = os.path.join(job_path, f"{job_id}.json")
    if not os.path.exists(job_file):
        return None

    try:
        with open(job_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading job data: {e}")
        return None


def list_saved_jobs():
    """
    List all saved job data files

    Returns:
        list: List of job IDs
    """
    jobs = []
    for filename in os.listdir(job_path):
        if filename.endswith('.json'):
            job_id = filename.replace('.json', '')
            jobs.append(job_id)
    return jobs


async def process_queue():
    """
    Process the next job in the queue

    This function will check if there's already a job running,
    and if not, will start the next job in the queue.
    """
    global running_job_id, job_queue

    # Acquire lock to prevent multiple instances running
    async with job_processing_lock:
        if running_job_id is not None:
            # Already processing a job
            return

        if not job_queue:
            # No jobs in queue
            return

        # Get the next job
        running_job_id = job_queue.pop(0)

        # Update queue positions
        update_queue_positions()

    try:
        # Process the job
        await run_job(running_job_id)
    finally:
        # Clear running job ID and process next job
        running_job_id = None
        # If more jobs in queue, process the next one
        if job_queue:
            await asyncio.create_task(process_queue())


async def run_job(job_id: str):
    """
    Run a specific job by ID

    Args:
        job_id: The job ID to run
    """

    # Load job settings
    if job_id not in job_statuses:
        job_data = load_job_data(job_id)
        if not job_data:
            print(f"Job {job_id} not found")
            return

        job_status = JobStatus(job_id)
        job_status.__dict__.update(job_data)
        job_statuses[job_id] = job_status

    job_status = job_statuses[job_id]
    job_settings = job_status.job_settings

    if not job_settings:
        print(f"Job {job_id} has no settings")
        job_status.status = "failed"
        job_status.message = "No job settings found"
        return

    try:
        # Update status
        job_status.status = "running"
        job_status.progress = 0
        job_status.message = "Starting..."

        # Load models

        # Determine which module to use based on job_settings keys
        module_name = next(iter(job_settings.keys()), "framepack")
        module_settings = job_settings.get(module_name, {})

        if module_name == "framepack":
            # Extract segments for processing
            segments = []
            for segment in module_settings.get("segments", []):
                segments.append({
                    "image_path": segment.get("image_path"),
                    "prompt": segment.get("prompt", ""),
                    "duration": segment.get("duration", 1.0)
                })

            # Start the job in a separate thread
            from modules.framepack.module import worker_multi_segment

            await asyncio.to_thread(
                worker_multi_segment,
                job_id=job_id,
                segments=segments,
                global_prompt=module_settings.get("global_prompt", ""),
                n_prompt=module_settings.get("negative_prompt", ""),
                seed=module_settings.get("seed", 31337),
                steps=module_settings.get("steps", 25),
                cfg=1.0,  # Fixed parameter
                gs=module_settings.get("guidance_scale", 10.0),
                rs=0.0,  # Fixed parameter
                gpu_memory_preservation=module_settings.get("gpu_memory_preservation", 6.0),
                use_teacache=module_settings.get("use_teacache", True),
                mp4_crf=module_settings.get("mp4_crf", 16),
                enable_adaptive_memory=module_settings.get("enable_adaptive_memory", True),
                resolution=module_settings.get("resolution", 640)
            )
        else:
            # Add support for other modules here if needed
            job_status.status = "failed"
            job_status.message = f"Unknown module type: {module_name}"

    except Exception as e:
        print(f"Error running job {job_id}: {e}")
        import traceback
        traceback.print_exc()

        # Update job status
        job_status.status = "failed"
        job_status.message = f"Error: {str(e)}"


def update_queue_positions():
    """Update queue position numbers for all jobs in the queue"""
    for i, job_id in enumerate(job_queue):
        if job_id in job_statuses:
            job_statuses[job_id].queue_position = i + 1  # +1 because 0 = running
            save_job_data(job_id, job_statuses[job_id].to_dict())


def verify_job_images(job_data):
    """
    Check if all images in a job exist

    Args:
        job_data: Dictionary of job data

    Returns:
        tuple: (is_valid, invalid_images) where invalid_images is a list of missing image paths
    """
    if not job_data or 'segments' not in job_data:
        return False, []

    missing_images = []

    for segment in job_data['segments']:
        image_path = segment.get('image_path')
        if not image_path:
            missing_images.append("Missing path")
            continue

        # Normalize path for cross-platform compatibility
        norm_path = os.path.normpath(image_path)

        # Handle file:// protocol if present
        if norm_path.startswith('file://'):
            norm_path = norm_path[7:]

        # Check if the file exists
        if not os.path.exists(norm_path) or not os.path.isfile(norm_path):
            missing_images.append(image_path)

    return len(missing_images) == 0, missing_images


def register_api_endpoints(app):
    """Register job queue related API endpoints"""
    api_tag = __name__.split(".")[-1].title().replace("_", " ")

    @app.post("/api/requeue_job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def requeue_job(job_id: str, position: int = None):
        """
        Re-queue a job without running it immediately
        
        Args:
            job_id: The unique job identifier
            position: Optional position in queue (0-based, None = end of queue)
        
        # Validate job_settings and segments to ensure they are properly structured
        if "job_settings" in cleaned_data and cleaned_data["job_settings"] is not None:
            if "segments" in cleaned_data["job_settings"]:
                print(f"Saving job {job_id} with {len(cleaned_data['job_settings']['segments'])} segments in job settings")
            else:
                print(f"Warning: job_settings for {job_id} is missing 'segments' key")
                
                # If segments exist in the main job data but not in job_settings, add them
                if "segments" in cleaned_data and cleaned_data["segments"]:
                    cleaned_data["job_settings"]["segments"] = []
                    # Create basic segment configurations
                    for segment_path in cleaned_data["segments"]:
                        cleaned_data["job_settings"]["segments"].append({
                            "image_path": segment_path,
                            "prompt": "",
                            "duration": 1.0
                        })
                    print(f"Fixed job_settings by adding {len(cleaned_data['segments'])} segment configurations")
                
        # Ensure segments are saved
        if "segments" in cleaned_data:
            print(f"Saving job {job_id} with {len(cleaned_data['segments'])} segments")
        
        Returns:
            Success message or error
        """

        # Try to load the job data
        job_data = None
        if job_id in job_statuses:
            job_data = job_statuses[job_id].to_dict()
        else:
            job_data = load_job_data(job_id)

        if not job_data:
            return ErrorResponse(error="Job not found")

        # Check if the job has settings
        job_settings = job_data.get("job_settings")
        if not job_settings:
            return ErrorResponse(error="Cannot requeue job: missing settings")

        # Verify images still exist
        is_valid, missing_images = verify_job_images(job_settings)
        if not is_valid:
            return ErrorResponse(
                error=f"Cannot requeue job: {len(missing_images)} missing images: {', '.join(missing_images[:3])}")

        # Create job status if not in memory
        if job_id not in job_statuses:
            job_status = JobStatus(job_id)
            job_status.__dict__.update(job_data)
            job_statuses[job_id] = job_status
        else:
            job_status = job_statuses[job_id]

        # Update job status
        job_status.status = "queued"
        job_status.progress = 0
        job_status.message = "Waiting in queue"
        save_job_data(job_id, job_status.to_dict())

        # Add to queue
        add_to_queue(job_id, position)

        return {
            "success": True,
            "message": f"Job {job_id} has been queued",
            "queue_position": job_status.queue_position
        }

    @app.post("/api/cancel_all_jobs", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def cancel_all_jobs():
        """
        Cancel all running and queued jobs
        
        Returns:
            Success message with count of cancelled jobs
        """
        global job_queue, running_job_id

        cancelled_count = 0

        # Cancel running job if any
        if running_job_id:
            if running_job_id in job_statuses:
                job_status = job_statuses[running_job_id]
                job_status.status = "cancelled"
                job_status.message = "Job cancelled by user"
                save_job_data(running_job_id, job_status.to_dict())
                cancelled_count += 1
            running_job_id = None

        # Cancel all queued jobs
        for job_id in list(job_queue):  # Use a copy since we'll modify the queue
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
                job_status.status = "cancelled"
                job_status.message = "Job cancelled by user"
                save_job_data(job_id, job_status.to_dict())
                cancelled_count += 1

        # Clear the queue
        job_queue.clear()

        return {
            "success": True,
            "message": f"Cancelled {cancelled_count} jobs",
            "cancelled_count": cancelled_count
        }

    @app.post("/api/update_queue_order", response_model=Union[List[JobStatusResponse], ErrorResponse], tags=[api_tag])
    async def update_queue_order(job_ids: List[str]):
        """
        Update the order of jobs in the queue
        
        Args:
            job_ids: List of job IDs in the desired order
        
        Returns:
            Updated list of job statuses
        """
        global job_queue

        # Validate all job IDs exist
        missing_jobs = []
        for job_id in job_ids:
            if job_id not in job_statuses and not load_job_data(job_id):
                missing_jobs.append(job_id)

        if missing_jobs:
            return ErrorResponse(error=f"Some jobs not found: {', '.join(missing_jobs)}")

        # Update the queue
        job_queue = job_ids.copy()

        # Update all queue positions
        update_queue_positions()

        # Get updated job statuses
        updated_jobs = []
        for job_id in job_queue:
            try:
                if job_id in job_statuses:
                    updated_jobs.append(JobStatusResponse(**job_statuses[job_id].to_dict()))
                else:
                    job_data = load_job_data(job_id)
                    if job_data:
                        updated_jobs.append(JobStatusResponse(**job_data))
            except Exception as e:
                print(f"Error getting job status for {job_id}: {e}")

        return updated_jobs

    @app.get("/api/job_status/{job_id}", response_model=Union[JobStatusResponse, ErrorResponse], tags=[api_tag])
    async def get_job_status(job_id: str):
        """
        Get the status of a video generation job

        Args:
            job_id: The unique job identifier

        Returns:
            The job status information
        """
        # First check if job is in memory
        if job_id in job_statuses:
            status = job_statuses[job_id]
            # Verify that any images are still valid
            job_settings = status.job_settings
            if job_settings and "segments" in job_settings:
                is_valid, missing_images = verify_job_images(job_settings)
                status.is_valid = is_valid
                status.missing_images = missing_images
            return JobStatusResponse(**status.to_dict())

        # If not in memory, try to load from disk
        job_data = load_job_data(job_id)
        if not job_data:
            return ErrorResponse(error="Job not found")

        # Create a JobStatus object from the saved data
        status = JobStatus(job_id)
        status.status = job_data.get("status", "unknown")
        status.progress = job_data.get("progress", 0)
        status.message = job_data.get("message", "")
        status.result_video = job_data.get("result_video")
        status.segments = job_data.get("segments", [])
        status.current_latents = job_data.get("current_latents")
        status.job_settings = job_data.get("job_settings")

        # Verify images are still valid
        if status.job_settings and "segments" in status.job_settings:
            is_valid, missing_images = verify_job_images(status.job_settings)
            status.is_valid = is_valid
            status.missing_images = missing_images

        # Add to in-memory cache
        job_statuses[job_id] = status

        return JobStatusResponse(**status.to_dict())

    @app.post("/api/save_job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def save_job(
            job_id: str,
            job_data: SaveJobRequest
    ):
        """
        Save a job to disk
        
        Args:
            job_id: The unique job identifier
            job_data: The job data to save
            
        Returns:
            Success message or error
        """
        try:
            # Validate that the job exists
            existing_job = None

            if job_id in job_statuses:
                existing_job = job_statuses[job_id]
            else:
                existing_job_data = load_job_data(job_id)
                if existing_job_data:
                    # Create a JobStatus object from the saved data
                    existing_job = JobStatus(job_id)
                    existing_job.__dict__.update(existing_job_data)

            if not existing_job and not load_job_data(job_id):
                # If no existing job, we'll create a new one with this ID
                print(f"Creating new job with ID {job_id}")
                existing_job = JobStatus(job_id)

            # Ensure job_id in path matches job_id in request
            if job_data.job_id != job_id:
                return ErrorResponse(error="Job ID mismatch between URL and request body")

            # Update job_status with data from request
            if existing_job:
                existing_job.status = "saved"
                existing_job.progress = job_data.progress
                existing_job.message = job_data.message

                # Only update result_video if it's provided and not empty
                if job_data.result_video:
                    existing_job.result_video = job_data.result_video

                # Only update segments if they're provided and not empty
                if job_data.segments:
                    existing_job.segments = job_data.segments

                # Update remaining fields
                if job_data.current_latents:
                    existing_job.current_latents = job_data.current_latents

                existing_job.is_valid = job_data.is_valid
                existing_job.missing_images = job_data.missing_images

                # Update job settings
                if job_data.job_settings:
                    existing_job.job_settings = job_data.job_settings

                existing_job.queue_position = job_data.queue_position

                # Update or add to in-memory cache
                job_statuses[job_id] = existing_job

                # Save to disk
                save_job_data(job_id, existing_job)
            else:
                # Save directly from request data
                save_job_data(job_id, job_data)

            return {
                "success": True,
                "message": f"Job {job_id} has been saved"
            }
        except Exception as e:
            print(f"Error saving job: {e}")
            import traceback
            traceback.print_exc()
            return ErrorResponse(error=f"Error saving job: {str(e)}")

    @app.get("/api/check_file_exists", response_model=dict, tags=[api_tag])
    async def check_file_exists(path: str):
        """
        Check if a file exists on the server
        
        Args:
            path: The file path to check
            
        Returns:
            A dictionary with exists: True/False
        """
        # Clean the path to prevent directory traversal attacks
        clean_path = os.path.normpath(path)

        # If path starts with file:// protocol, remove it
        if clean_path.startswith('file://'):
            clean_path = clean_path[7:]

        # Check if the file exists
        exists = os.path.exists(clean_path) and os.path.isfile(clean_path)

        return {"exists": exists}

    @app.get("/api/result_video/{job_id}", tags=[api_tag])
    async def get_result_video(job_id: str):
        """
        Get the video file for a completed job

        Args:
            job_id: The unique job identifier

        Returns:
            The video file
        """
        # First check in-memory cache
        if job_id in job_statuses and job_statuses[job_id].result_video:
            video_path = job_statuses[job_id].result_video
        else:
            # Try to load from disk
            job_data = load_job_data(job_id)
            if not job_data or not job_data.get("result_video"):
                raise HTTPException(status_code=404, detail="Video not found")
            video_path = job_data.get("result_video")

        # Verify the file exists
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            raise HTTPException(status_code=404, detail="Video file missing from disk")

        return FileResponse(
            path=video_path,
            filename=f"{job_id}.mp4",
            media_type="video/mp4"
        )

    @app.get("/api/list_jobs", response_model=List[JobStatusResponse], tags=[api_tag])
    async def list_jobs():
        """
        List all jobs (both active in-memory jobs and saved jobs)

        Returns:
            A list of all job statuses
        """
        # Get all job IDs from in-memory and saved jobs
        all_job_ids = set(job_statuses.keys()).union(set(list_saved_jobs()))

        # Collect all job statuses
        all_jobs = []
        for job_id in all_job_ids:
            try:
                # Get job status (this will fetch from memory or disk)
                job_resp = await get_job_status(job_id)
                if not isinstance(job_resp, ErrorResponse):  # Only include valid jobs
                    all_jobs.append(job_resp)
            except Exception as e:
                print(f"Error getting job status for {job_id}: {e}")

        return all_jobs

    @app.post("/api/reload_job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def reload_job(job_id: str):
        """
        Reload a saved job to get its settings and configuration

        Args:
            job_id: The unique job identifier

        Returns:
            The job data for reloading into the UI
        """
        # Try to load the job data
        job_data = load_job_data(job_id)
        if not job_data:
            return ErrorResponse(error="Job not found")

        # Check if the job has settings
        job_settings = job_data.get("job_settings")
        if not job_settings:
            return ErrorResponse(error="Job settings not found")

        # Verify images still exist
        is_valid, missing_images = verify_job_images(job_settings)

        return {
            "job_id": job_id,
            "is_valid": is_valid,
            "missing_images": missing_images,
            "job_settings": job_settings
        }

    @app.post("/api/rerun_job/{job_id}", response_model=Union[GenerateResponse, ErrorResponse], tags=[api_tag])
    async def rerun_job(
            job_id: str,
            background_tasks: BackgroundTasks
    ):
        """
        Re-run a completed or failed job

        Args:
            job_id: The unique job identifier

        Returns:
            new_job_id: A new job ID for the rerun
        """
        from modules.framepack.diffusers_helper.utils import generate_timestamp

        # Try to load the job data
        job_data = load_job_data(job_id)
        if not job_data:
            return ErrorResponse(error="Job not found")

        # Check if the job has settings
        job_settings = job_data.get("job_settings")
        if not job_settings:
            return ErrorResponse(error="Cannot rerun job: missing settings")

        # Verify images still exist
        is_valid, missing_images = verify_job_images(job_settings)
        if not is_valid:
            return ErrorResponse(
                error=f"Cannot rerun job: {len(missing_images)} missing images: {', '.join(missing_images[:3])}")

        # Create a new job ID
        new_job_id = generate_timestamp()

        # Create request data from job settings
        try:
            # Copy all fields from the original job settings
            segments = []
            for segment in job_settings.get("segments", []):
                segments.append(SegmentConfig(
                    image_path=segment.get("image_path"),
                    prompt=segment.get("prompt", ""),
                    duration=segment.get("duration", 1.0)
                ))

            request_data = VideoRequest(
                global_prompt=job_settings.get("global_prompt", ""),
                negative_prompt=job_settings.get("negative_prompt", ""),
                segments=segments,
                seed=job_settings.get("seed", 31337),
                steps=job_settings.get("steps", 25),
                guidance_scale=job_settings.get("guidance_scale", 10.0),
                use_teacache=job_settings.get("use_teacache", True),
                enable_adaptive_memory=job_settings.get("enable_adaptive_memory", True),
                resolution=job_settings.get("resolution", 640),
                mp4_crf=job_settings.get("mp4_crf", 16),
                gpu_memory_preservation=job_settings.get("gpu_memory_preservation", 6.0)
            )

            # Create the job status
            job_status = JobStatus(new_job_id)
            job_status.set_job_settings(request_data.model_dump())
            job_statuses[new_job_id] = job_status

            # Save to disk
            save_job_data(new_job_id, job_status.to_dict())

            # Add job to the queue
            add_to_queue(new_job_id)

            return GenerateResponse(job_id=new_job_id)

        except Exception as e:
            print(f"Error rerunning job: {e}")
            return ErrorResponse(error=f"Error rerunning job: {str(e)}")

    @app.delete("/api/job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def delete_job(job_id: str, request: DeleteJobRequest = None):
        """
        Delete a job and its associated files

        Args:
            job_id: The unique job identifier
            request: Optional request with delete_images flag

        Returns:
            Success message or error
        """
        job_in_memory = job_id in job_statuses
        job_data = None

        # Try to load from disk if not in memory
        if not job_in_memory:
            job_data = load_job_data(job_id)
            if not job_data:
                return ErrorResponse(error="Job not found")
        else:
            job_data = job_statuses[job_id].to_dict()

        # Initialize deleted images counter
        deleted_images = 0

        # Collect images to potentially delete
        image_paths_to_check = set()

        # Add images from job settings
        if job_data.get("job_settings") and "segments" in job_data["job_settings"]:
            for segment in job_data["job_settings"]["segments"]:
                if "image_path" in segment and os.path.exists(segment["image_path"]):
                    image_paths_to_check.add(segment["image_path"])

        # Delete result video if it exists
        if job_data.get("result_video") and os.path.exists(job_data["result_video"]):
            try:
                os.remove(job_data["result_video"])
            except Exception as e:
                print(f"Failed to delete video file: {e}")

        # Delete segment videos if they exist
        for segment in job_data.get("segments", []):
            if os.path.exists(segment):
                try:
                    os.remove(segment)
                except Exception as e:
                    print(f"Failed to delete segment file: {e}")

        # Delete job JSON file
        job_file = os.path.join(job_path, f"{job_id}.json")
        if os.path.exists(job_file):
            try:
                os.remove(job_file)
            except Exception as e:
                print(f"Failed to delete job file: {e}")

        # Delete latent previews
        for filename in os.listdir(upload_path):
            if filename.startswith(f"{job_id}_latent_"):
                try:
                    os.remove(os.path.join(upload_path, filename))
                except Exception as e:
                    print(f"Failed to delete latent preview: {e}")

        # Remove job from in-memory statuses
        if job_in_memory:
            del job_statuses[job_id]

        # If requested, delete image files only if not used by other jobs
        if request and request.delete_images and image_paths_to_check:
            # Get all image paths from all jobs
            all_image_paths = set()
            for other_job_id in list_saved_jobs():
                if other_job_id == job_id:  # Skip the job we're deleting
                    continue

                other_job_data = load_job_data(other_job_id)
                if other_job_data and other_job_data.get("job_settings") and "segments" in other_job_data[
                    "job_settings"]:
                    for segment in other_job_data["job_settings"]["segments"]:
                        if "image_path" in segment:
                            all_image_paths.add(segment["image_path"])

            # Delete images that are not used by any other jobs
            for image_path in image_paths_to_check:
                if image_path not in all_image_paths:
                    try:
                        if os.path.exists(image_path) and os.path.isfile(image_path):
                            os.remove(image_path)
                            deleted_images += 1
                    except Exception as e:
                        print(f"Failed to delete image file {image_path}: {e}")

        return {
            "success": True,
            "message": f"Job {job_id} deleted",
            "deleted_images": deleted_images
        }

    @app.post("/api/cancel_job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def cancel_job(job_id: str):
        """
        Cancel a running job

        Args:
            job_id: The unique job identifier

        Returns:
            Success message or error
        """
        # Check if job exists
        if job_id not in job_statuses and not load_job_data(job_id):
            return ErrorResponse(error="Job not found")

        try:
            # Get the job status
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
            else:
                job_data = load_job_data(job_id)
                if not job_data:
                    return ErrorResponse(error="Job not found")
                job_status = JobStatus(job_id)
                job_status.__dict__.update(job_data)
                job_statuses[job_id] = job_status

            # Only cancel if the job is running or queued
            if job_status.status not in ["running", "queued"]:
                return ErrorResponse(error=f"Cannot cancel job with status '{job_status.status}'")

            # Update job status
            job_status.status = "cancelled"
            job_status.message = "Job cancelled by user"

            # Save the updated status to disk
            save_job_data(job_id, job_status.to_dict())

            # Note: The actual cancellation of the running process happens in the worker functions
            # They periodically check the job status and will detect the cancelled state

            # Clean up any temporary preview files
            latent_preview = os.path.join(upload_path, f"{job_id}_latent.jpg")
            if os.path.exists(latent_preview):
                try:
                    os.remove(latent_preview)
                except Exception as e:
                    print(f"Failed to delete latent preview: {e}")

            return {
                "success": True,
                "message": f"Job {job_id} has been cancelled"
            }

        except Exception as e:
            print(f"Error cancelling job: {e}")
            return ErrorResponse(error=f"Failed to cancel job: {str(e)}")
