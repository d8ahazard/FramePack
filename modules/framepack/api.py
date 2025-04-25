import os

from fastapi import BackgroundTasks, Body
from starlette.responses import JSONResponse

from datatypes.datatypes import GenerateResponse, VideoRequest, JobStatus, SaveJobRequest
from modules.framepack.diffusers_helper.utils import generate_timestamp
from modules.framepack.module import load_models
from handlers.job_queue import add_to_queue, save_job_data, job_statuses
from handlers.path import job_path


def register_api_endpoints(app):
    api_tag = __name__.split(".")[-2].title().replace("_", " ")

    @app.post("/api/generate_video", response_model=GenerateResponse, tags=[api_tag])
    async def generate_video(
            background_tasks: BackgroundTasks,
            request_data: VideoRequest
    ):
        """
        Generate a video from a series of images with prompts

        Args:
            request_data: VideoRequest object containing all generation parameters

        Returns:
            job_id: A unique identifier for the job
        """

        job_id = generate_timestamp()
        job_status = JobStatus(job_id, request_data.job_name)

        # Store original job settings for later use as a dictionary with module name as key
        module_settings = request_data.dict()
        job_settings = {"framepack": module_settings}
        job_status.set_job_settings(job_settings)

        # Save job to in-memory cache
        job_statuses[job_id] = job_status

        # Save job data to disk
        job_data = job_status.to_dict()
        save_job_data(job_id, job_data)

        # Load models
        load_models()

        # Log the request for debugging
        print(f"Video generation request received:")
        if request_data.job_name:
            print(f"  Job name: {request_data.job_name}")
        print(f"  Global prompt: {request_data.global_prompt[:50]}..." if len(
            request_data.global_prompt) > 50 else f"  Global prompt: {request_data.global_prompt}")
        print(f"  Negative prompt: {request_data.negative_prompt[:50]}..." if len(
            request_data.negative_prompt) > 50 else f"  Negative prompt: {request_data.negative_prompt}")
        print(f"  Number of segments: {len(request_data.segments)}")
        print(f"  Resolution: {request_data.resolution}")
        print(f"  Include last frame: {request_data.include_last_frame}")

        # Process segments
        segments = []
        for i, segment in enumerate(request_data.segments):
            # Verify that the image path exists
            if not os.path.exists(segment.image_path):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Image file not found for segment {i + 1}: {segment.image_path}"}
                )

            print(f"  Segment {i + 1}: image_path={segment.image_path}")
            print(f"    Prompt: {segment.prompt[:50]}..." if len(segment.prompt) > 50 else f"    Prompt: {segment.prompt}")
            print(f"    Duration: {segment.duration}s")

            segments.append({
                "image_path": segment.image_path,
                "prompt": segment.prompt,
                "duration": segment.duration
            })

        # Add job to the queue system
        add_to_queue(job_id)

        return GenerateResponse(job_id=job_id)
