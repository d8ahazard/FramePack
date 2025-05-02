import asyncio
import json
import logging
import os
import time
import traceback
from typing import List, Union
from typing import Optional

import torch
from fastapi import BackgroundTasks, HTTPException
from starlette.responses import FileResponse

from datatypes.datatypes import GenerateResponse, JobStatus, SegmentConfig, \
    DeleteJobRequest
from datatypes.datatypes import JobStatusResponse, ErrorResponse
from datatypes.datatypes import SaveJobRequest
from handlers.path import job_path
from handlers.path import upload_path

# Configure logging
logger = logging.getLogger(__name__)

# Job queue and processing state
job_queue = []
running_job_ids = set()  # Set of currently running job IDs
job_processing_lock = asyncio.Lock()

# Global job tracking
job_statuses = {}

async def startup_event():
    clear_running_jobs()


async def shutdown_event():
    global job_queue, running_job_ids, job_statuses
    for job_id in running_job_ids:
        job_status = job_statuses.get(job_id)
        if job_status:
            job_status.status = "cancelled"
            job_status.message = "Job was interrupted"
            save_job_data(job_id, job_status.to_dict())
    job_queue.clear()
    running_job_ids.clear()
    job_statuses.clear()
    clear_running_jobs()
    logger.info("Shutdown event complete for job queue")


# Get the number of available GPUs
def get_available_gpu_count():
    """
    Returns the number of available CUDA GPUs.

    Returns:
        int: Number of available GPUs, minimum 1 if CUDA is available
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, defaulting to 1 virtual GPU")
        return 1

    return torch.cuda.device_count()


# Run this at app start in case we terminated/failed
def clear_running_jobs():
    """Reset any jobs that were running when the server was stopped."""
    global running_job_ids

    # Get all saved jobs
    saved_jobs = list_saved_jobs()
    # Check if any jobs are running
    for job_id in saved_jobs:
        job_data = load_job_data(job_id)
        if job_data:
            # Set the current latents to None
            job_data["current_latents"] = None
            # Check if the job is running
            if job_data.get("status").lower() == "running" or job_data.get("status").lower() == "cancelled":
                # If so, set it to failed
                job_data["status"] = "cancelled"
                job_data["message"] = "Job was interrupted"
                job_data["progress"] = 0
            
            if job_data.get("status").lower() == "saved":
                job_data["message"] = "Job Saved"
                job_data["progress"] = 0
            
            if job_data.get("status").lower() == "completed":
                job_data["message"] = "Job Completed"
                job_data["progress"] = 100
            
            save_job_data(job_id, job_data)

    # Clear running jobs set
    running_job_ids.clear()


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

    # Only start processing if nothing is running
    if not running_job_ids:
        # Start queue processing if needed
        asyncio.create_task(process_queue())
    # Otherwise, let the running jobs handle continuing the queue


def save_job_data(job_id, data_dict):
    """Save job data to file"""
    # Create job path if not exists
    job_file = os.path.join(job_path, f"{job_id}.json")
    os.makedirs(os.path.dirname(job_file), exist_ok=True)

    # For compatibility: convert different types of input to dict
    if isinstance(data_dict, JobStatus):
        data_dict = data_dict.to_dict()
    elif hasattr(data_dict, "model_dump"):
        # Handle Pydantic models
        data_dict = data_dict.model_dump()
    elif hasattr(data_dict, "dict"):
        # Handle older Pydantic models
        data_dict = data_dict.dict()

    # Ensure job_id is consistent
    if isinstance(data_dict, dict):
        data_dict["job_id"] = job_id

        # Add timestamp if missing
        if "created_timestamp" not in data_dict:
            data_dict["created_timestamp"] = int(time.time())

        # Handle SegmentConfig objects in job_settings
        if 'job_settings' in data_dict and data_dict['job_settings']:
            job_settings = data_dict['job_settings']
            new_settings = {}
            for module_name, module_settings in job_settings.items():
                if 'segments' in module_settings:
                    if len(module_settings['segments']) > 0:
                        if isinstance(module_settings['segments'][0], SegmentConfig):
                            module_settings['segments'] = [segment.model_dump() for segment in
                                                           module_settings['segments']]
                new_settings[module_name] = module_settings
            data_dict['job_settings'] = new_settings

    try:
        with open(job_file, "w") as f:
            json.dump(data_dict, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving job data: {e}")
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
        logger.error(f"Error queuing job update for broadcast: {e}")
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
    Process jobs in the queue based on available GPUs.

    This function will check how many GPUs are available and start
    multiple jobs concurrently if possible.
    """
    global running_job_ids, job_queue

    # Acquire lock to prevent multiple instances running
    async with job_processing_lock:
        # Get the number of available GPUs
        max_concurrent_jobs = get_available_gpu_count()
        logger.info(f"Available GPUs: {max_concurrent_jobs}")

        # Check if we can start more jobs
        available_slots = max_concurrent_jobs - len(running_job_ids)

        if available_slots <= 0:
            # Already running maximum number of jobs
            logger.debug("Maximum number of concurrent jobs already running")
            return

        if not job_queue:
            # No jobs in queue
            logger.debug("No jobs in queue to process")
            return

        # Start as many jobs as we have available slots
        jobs_to_start = min(available_slots, len(job_queue))
        logger.info(f"Starting {jobs_to_start} new job(s)")

        jobs_started = []
        
        for _ in range(jobs_to_start):
            if not job_queue:
                break

            # Get the next job
            job_id = job_queue.pop(0)
            
            # Verify job data before starting
            job_data = load_job_data(job_id)
            if not job_data:
                logger.error(f"Job {job_id} data not found when trying to start")
                continue
                
            # Add to running jobs set
            running_job_ids.add(job_id)
            jobs_started.append(job_id)

            # Update status to running
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
                job_status.status = "running"
                job_status.message = "Starting job..."
                save_job_data(job_id, job_status.to_dict())

        # Update queue positions
        update_queue_positions()
        
        # Start the jobs outside the loop to reduce lock holding time
        for job_id in jobs_started:
            # Start the job in a separate task
            asyncio.create_task(run_job_and_process_next(job_id))


async def run_job_and_process_next(job_id: str):
    """
    Run a job and then process the next job in the queue.

    Args:
        job_id: The job ID to run
    """
    global running_job_ids

    try:
        # Process the job
        await run_job(job_id)
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())

        # Update job status to failed if it's still in the running state
        # Use lock to avoid race conditions
        async with job_processing_lock:
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
                if job_status.status == "running":
                    job_status.status = "failed"
                    job_status.message = f"Job failed with error: {str(e)}"
                    save_job_data(job_id, job_status.to_dict())
    finally:
        # Use lock for cleanup operations
        async with job_processing_lock:
            # Remove from running jobs set
            running_job_ids.discard(job_id)

            # Process next job if there are more in the queue
            if job_queue:
                # Don't await here to avoid blocking
                asyncio.create_task(process_queue())


async def run_job(job_id: str):
    """
    Run a specific job by ID

    Args:
        job_id: The job ID to run
    """
    import importlib

    # Load job settings
    if job_id not in job_statuses:
        job_data = load_job_data(job_id)
        if not job_data:
            logger.error(f"Job {job_id} not found")
            return

        job_status = JobStatus(job_id)
        job_status.__dict__.update(job_data)
        job_statuses[job_id] = job_status

    job_status = job_statuses[job_id]
    job_settings = job_status.job_settings

    if not job_settings:
        logger.error(f"Job {job_id} has no settings")
        job_status.status = "failed"
        job_status.message = "No job settings found"
        save_job_data(job_id, job_status)
        return

    try:
        # Update status
        job_status.status = "running"
        job_status.progress = 0
        job_status.message = "Starting..."
        save_job_data(job_id, job_status)
        logger.info(f"Starting job {job_id}")

        # TODO: Make this determined by our modules, eventually
        module_order = ["framepack", "facefusion"]
        # Look for module keys in job_settings
        for module_name in module_order:
            module_settings = job_settings.get(module_name)
            if not module_settings:
                continue

            try:
                # Import the module dynamically
                module_path = f"modules.{module_name}.module"
                # Get the base "package" name
                module = importlib.import_module(module_path)
                process_func = None
                request_type = None
                # Check if the module has a process function
                if hasattr(module, "process"):
                    logger.info(f"Calling {module_name}.process with settings")
                    # Get the type of the first parameter of the process function
                    process_func = getattr(module, "process")

                    if hasattr(process_func, "__annotations__") and "request" in process_func.__annotations__:
                        request_type = process_func.__annotations__["request"]
                        logger.info(f"Request type: {request_type}")

                # Add job_id to the settings if not present
                if isinstance(module_settings, dict) and "job_id" not in module_settings:
                    module_settings["job_id"] = job_id

                if process_func and request_type:
                    if "segments" in module_settings:
                        module_settings["segments"] = [SegmentConfig(**segment) for segment in
                                                       module_settings["segments"]]

                    request_instance = request_type(**module_settings)

                    # Run the worker function directly
                    await asyncio.to_thread(
                        process_func,
                        request_instance
                    )
                else:
                    job_status.status = "failed"
                    job_status.message = f"Module {module_name} does not have a process function or request type"
                    save_job_data(job_id, job_status)
            except Exception as e:
                print(f"Error processing module {module_name}: {e}")
                import traceback
                traceback.print_exc()
                job_status.status = "failed"
                job_status.message = f"Error in {module_name} module: {str(e)}"
                save_job_data(job_id, job_status)

    except Exception as e:
        print(f"Error running job {job_id}: {e}")
        import traceback
        traceback.print_exc()

        # Update job status
        job_status.status = "failed"
        job_status.message = f"Error: {str(e)}"
        save_job_data(job_id, job_status)


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
    if not job_data:
        return False, []

    missing_images = []

    # Check for different job settings structures
    segments = None

    # Direct segments structure
    if 'segments' in job_data:
        segments = job_data['segments']
    # Nested framepack structure
    elif 'framepack' in job_data and 'segments' in job_data['framepack']:
        segments = job_data['framepack']['segments']

    if not segments:
        return False, []

    for segment in segments:
        image_path = segment.get('image_path')
        seg_path = image_path.replace("/uploads/", "") if "/" in image_path else image_path.replace("\\uploads\\", "")
        image_path = os.path.join(upload_path, seg_path)
        logger.info(f"Checking segment path: {image_path}")
        if not image_path:
            missing_images.append("Missing path")
            continue

        # Check if the file exists
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            missing_images.append(image_path)

    return len(missing_images) == 0, missing_images


def register_api_endpoints(app):
    """Register job queue related API endpoints"""
    api_tag = __name__.split(".")[-1].title().replace("_", " ")

    @app.post("/api/requeue_job/{job_id}", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def requeue_job(job_id: str, position: int = None, background_tasks: BackgroundTasks = None):
        """
        Re-queue a job (backward compatibility endpoint)

        Args:
            job_id: The unique job identifier
            position: Optional position in queue (0-based, None = end of queue)

        Returns:
            Success message or error
        """
        # Forward to the run_job_endpoint which now handles all job running
        result = await run_job_endpoint(job_id, background_tasks)

        # If the result is a GenerateResponse, convert it to a dict with success message
        if isinstance(result, GenerateResponse):
            # Get the job status to return the queue position
            job_status = None
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
            else:
                job_data = load_job_data(job_id)
                if job_data:
                    job_status = JobStatus(job_id)
                    job_status.__dict__.update(job_data)

            queue_position = job_status.queue_position if job_status else 0

            return {
                "success": True,
                "message": f"Job {job_id} has been queued",
                "queue_position": queue_position
            }

        # If it's an error, just return it
        return result

    @app.post("/api/cancel_all_jobs", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def cancel_all_jobs():
        """
        Cancel all running and queued jobs

        Returns:
            Success message with count of cancelled jobs
        """
        global job_queue, running_job_ids

        cancelled_count = 0

        # Cancel all running jobs
        for job_id in list(running_job_ids):  # Use a copy since we'll modify the set
            if job_id in job_statuses:
                job_status = job_statuses[job_id]
                job_status.status = "cancelled"
                job_status.message = "Job cancelled by user"
                save_job_data(job_id, job_status.to_dict())
                cancelled_count += 1

        # Clear running jobs set
        running_job_ids.clear()
        logger.info(f"Cancelled {cancelled_count} running jobs")

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

    @app.post("/api/run_job/{job_id}", response_model=Union[GenerateResponse, ErrorResponse], tags=[api_tag])
    async def run_job_endpoint(
            job_id: str,
            background_tasks: BackgroundTasks
    ):
        """
        Run a job (whether new or existing)

        Args:
            job_id: The unique job identifier

        Returns:
            job_id: The job ID that was queued
        """
        # Try to load the job data
        job_data = load_job_data(job_id)
        if not job_data:
            return ErrorResponse(error="Job not found")

        # Check if the job has settings
        job_settings = job_data.get("job_settings")
        if not job_settings:
            return ErrorResponse(error="Cannot run job: missing settings")

        # Verify images still exist
        is_valid, missing_images = verify_job_images(job_settings)
        if not is_valid:
            print(f"Invalid job {job_id}, missing images: {missing_images}")
            return ErrorResponse(
                error=f"Cannot run job: {len(missing_images)} missing images: {', '.join(missing_images[:3])}")

        # Create or update job status - get lock for consistency
        async with job_processing_lock:
            job_status = job_statuses.get(job_id, JobStatus(job_id))

            # Update status
            job_status.status = "queued"
            job_status.progress = 0
            job_status.message = "Waiting in queue"
            job_status.result_video = ""  # Clear previous result

            # Ensure job settings are in the right format
            job_status.job_settings = job_settings

            # Save to memory cache
            job_statuses[job_id] = job_status

            # Save to disk
            save_job_data(job_id, job_status.to_dict())

            # Add job to the queue
            add_to_queue(job_id)

        return GenerateResponse(job_id=job_id)

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

        # # Add images from job settings
        # if job_data.get("job_settings") and "segments" in job_data["job_settings"]:
        #     for segment in job_data["job_settings"]["segments"]:
        #         if "image_path" in segment and os.path.exists(segment["image_path"]):
        #             image_paths_to_check.add(segment["image_path"])

        # # Delete result video if it exists
        # if job_data.get("result_video") and os.path.exists(job_data["result_video"]):
        #     try:
        #         os.remove(job_data["result_video"])
        #     except Exception as e:
        #         print(f"Failed to delete video file: {e}")

        # Delete segment videos if they exist
        for segment in job_data.get("segments", []):
            if os.path.exists(segment) and os.path.isfile(segment):
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
        global running_job_ids, job_queue

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

            # Remove from running jobs set if it's there
            if job_id in running_job_ids:
                running_job_ids.discard(job_id)
                logger.info(f"Removed job {job_id} from running jobs")

            # Remove from queue if it's there
            if job_id in job_queue:
                job_queue.remove(job_id)
                logger.info(f"Removed job {job_id} from queue")
                # Update queue positions
                update_queue_positions()

            # Note: The actual cancellation of the running process happens in the worker functions
            # They periodically check the job status and will detect the cancelled state

            # Clean up any temporary preview files
            latent_preview = os.path.join(upload_path, f"{job_id}_latent.jpg")
            if os.path.exists(latent_preview):
                try:
                    os.remove(latent_preview)
                except Exception as e:
                    logger.error(f"Failed to delete latent preview: {e}")

            return {
                "success": True,
                "message": f"Job {job_id} has been cancelled"
            }

        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            logger.error(traceback.format_exc())
            return ErrorResponse(error=f"Failed to cancel job: {str(e)}")
