import os
import time
import logging
from datetime import datetime
import traceback
from typing import Optional
from PIL import Image
import torch
import random
import io
import base64

from datatypes.datatypes import JobStatus
from handlers.job_queue import job_statuses, save_job_data
from handlers.model import check_download_model
from handlers.path import output_path, upload_path
from handlers.vram import gpu
from modules.wan.datatypes import WanJobSettings
from modules.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

# Import diffusers components for i2v
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel

# Import custom components for FLF2V
from modules.wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
try:
    from modules.wan.first_last_frame2video import WanFLF2V
except ImportError:
    WanFLF2V = None

logger = logging.getLogger(__name__)

# Model repository IDs for different Wan models
MODEL_REPOS = {
    "i2v-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "i2v-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    "flf2v-14B": "Wan-AI/Wan2.1-FLF2V-14B-720P",  # Using custom implementation
}

def update_status(job_id, message, status="running", progress: int = None, latent_preview: str = None, video_preview: str = None):
    """Update the job status in the queue"""
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    job_status.message = message
    if status is not None:
        job_status.status = status
    if latent_preview is not None:
        job_status.current_latents = latent_preview
    if video_preview is not None:
        video_url = f"/outputs/{os.path.basename(video_preview)}"
        job_status.result_video = video_url
        job_status.video_timestamp = time.time()
    if progress is not None:
        job_status.progress = progress

    # Persist updated job status
    save_job_data(job_id, job_status.to_dict())


def extend_prompt(prompt, method, model=None, target_lang="zh", image=None, seed=None):
    """Extend the prompt using specified method"""
    logger.info(f"Extending prompt using {method}...")
    
    if method == "dashscope":
        expander = DashScopePromptExpander(
            model_name=model, 
            is_vl=image is not None
        )
    elif method == "local_qwen":
        expander = QwenPromptExpander(
            model_name=model,
            is_vl=image is not None,
            device=0
        )
    else:
        raise ValueError(f"Unsupported prompt extension method: {method}")
    
    prompt_output = expander(
        prompt,
        tar_lang=target_lang,
        image=image,
        seed=seed
    )
    
    if not prompt_output.status:
        logger.info(f"Extending prompt failed: {prompt_output.message}")
        logger.info("Falling back to original prompt.")
        return prompt
    
    logger.info(f"Extended prompt: {prompt_output.prompt}")
    return prompt_output.prompt


def image_to_video(settings):
    """Generate video from an image using Wan2.1 I2V model with diffusers"""
    job_id = settings.job_id
    
    # Validate inputs
    if not settings.image:
        error_msg = "Image path is required for image-to-video generation"
        logger.error(error_msg)
        update_status(job_id, error_msg, status="failed", progress=100)
        raise ValueError(error_msg)
    
    # Determine resolution and model based on size
    height, width = map(int, settings.size.split('*'))
    max_area = height * width
    
    if max_area <= 832*480:
        model_id = MODEL_REPOS["i2v-14B-480P"]
        flow_shift = 3.0
    else:
        model_id = MODEL_REPOS["i2v-14B-720P"]
        flow_shift = 5.0
    
    update_status(job_id, f"Loading model: {model_id}", progress=10)
    
    # Handle image path - could be a data URL, relative path, or full path
    image_path = settings.image
    
    update_status(job_id, "Loading image...", progress=15)
    
    # Load the image
    try:
        # Handle data URLs directly
        if image_path.startswith('data:'):
            logger.info("Loading image from data URL")
            image = load_image(image_path)
        else:
            # For normal file paths, make sure we have the full path
            if not os.path.isabs(image_path):
                # Check if we need to extract the basename (web path)
                if image_path.startswith('/') or image_path.startswith('\\'):
                    image_path = os.path.basename(image_path)
                # Join with upload path
                image_path = os.path.join(upload_path, image_path)
            
            # Check if image exists
            if not os.path.exists(image_path):
                error_msg = f"Image not found at path: {image_path}"
                logger.error(error_msg)
                update_status(job_id, error_msg, status="failed", progress=100)
                raise FileNotFoundError(error_msg)
            
            # Load the image file
            image = load_image(image_path)
    except Exception as e:
        error_msg = f"Failed to load image: {str(e)}"
        logger.error(error_msg)
        update_status(job_id, error_msg, status="failed", progress=100)
        raise e
    
    # Process prompt if extension is enabled
    prompt = settings.prompt
    if settings.use_prompt_extend:
        update_status(job_id, "Extending prompt...", progress=20)
        prompt = extend_prompt(
            prompt=settings.prompt,
            method=settings.prompt_extend_method,
            model=settings.prompt_extend_model,
            target_lang=settings.prompt_extend_target_lang,
            image=image,
            seed=settings.base_seed if settings.base_seed >= 0 else random.randint(0, 2**32)
        )
    
    # Calculate dimensions
    aspect_ratio = image.height / image.width
    
    # Initialize the model components
    update_status(job_id, "Initializing model components...", progress=30)
    try:
        # Set up VAE and image encoder
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        
        # Set up scheduler and pipeline
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction', 
            use_flow_sigmas=True, 
            num_train_timesteps=1000, 
            flow_shift=settings.sample_shift if settings.sample_shift is not None else flow_shift
        )
        
        # Create pipeline and move to GPU
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, 
            vae=vae, 
            image_encoder=image_encoder, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )
        pipe.scheduler = scheduler
        pipe.to(gpu())
        
        # Calculate dimensions to maintain aspect ratio
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        i2v_height = round((max_area * aspect_ratio) ** 0.5) // mod_value * mod_value
        i2v_width = round((max_area / aspect_ratio) ** 0.5) // mod_value * mod_value
        
        # Resize image to calculated dimensions
        logger.info(f"Resizing image to dimensions: {i2v_width}x{i2v_height}")
        image = image.resize((i2v_width, i2v_height))
        
        # Generate the video
        update_status(job_id, "Generating video...", progress=40)
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=settings.negative_prompt,
            height=i2v_height,
            width=i2v_width,
            num_frames=settings.frame_num,
            guidance_scale=settings.sample_guide_scale,
            num_inference_steps=settings.sample_steps
        ).frames[0]
        
        # Save the video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_path, f"wan_i2v_{timestamp}.mp4")
        update_status(job_id, "Saving video...", progress=90)
        export_to_video(output, output_file, fps=settings.fps)
        
        update_status(job_id, "Video generation completed", status="completed", progress=100, video_preview=output_file)
        return output_file
        
    except Exception as e:
        error_msg = f"Error generating video: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        update_status(job_id, error_msg, status="failed", progress=100)
        raise e


def first_last_frame_to_video(request):
    """Generate video from first and last frames using Wan2.1 FLF2V model"""
    job_id = request.job_id
    
    # Validate inputs
    if not request.first_frame or not request.last_frame:
        error_msg = "First and last frame paths are required for FLF2V generation"
        logger.error(error_msg)
        update_status(job_id, error_msg, status="failed", progress=100)
        raise ValueError(error_msg)
    
    # Determine model and checkpoint directory
    model_id = MODEL_REPOS["flf2v-14B"]
    checkpoint_dir = check_download_model(model_id, module="Wan-AI")
    
    # Determine config based on settings
    config = WAN_CONFIGS.get('flf2v-14B')
    if not config:
        error_msg = "Missing config for FLF2V-14B"
        logger.error(error_msg)
        update_status(job_id, error_msg, status="failed", progress=100)
        raise ValueError(error_msg)
    
    try:
        # Update status
        update_status(job_id, "Loading FLF2V model...", progress=10)
        
        # Initialize the pipeline with default values for parameters that aren't in the request
        flf2v_pipeline = WanFLF2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            t5_fsdp=False,           # Default value, not in request
            dit_fsdp=False,          # Default value, not in request
            use_usp=False,           # Default value, not in request
            t5_cpu=request.t5_cpu,   # This one is in the request
            init_on_cpu=True,        # Default value
        )
        
        # Handle frame paths
        first_frame_path = request.first_frame
        last_frame_path = request.last_frame
        
        # Normalize paths if needed
        if not os.path.isabs(first_frame_path):
            if first_frame_path.startswith('/') or first_frame_path.startswith('\\'):
                first_frame_path = os.path.basename(first_frame_path)
            first_frame_path = os.path.join(upload_path, first_frame_path)
            
        if not os.path.isabs(last_frame_path):
            if last_frame_path.startswith('/') or last_frame_path.startswith('\\'):
                last_frame_path = os.path.basename(last_frame_path)
            last_frame_path = os.path.join(upload_path, last_frame_path)
        
        # Check if frames exist
        if not os.path.exists(first_frame_path):
            error_msg = f"First frame not found at path: {first_frame_path}"
            logger.error(error_msg)
            update_status(job_id, error_msg, status="failed", progress=100)
            raise FileNotFoundError(error_msg)
            
        if not os.path.exists(last_frame_path):
            error_msg = f"Last frame not found at path: {last_frame_path}"
            logger.error(error_msg)
            update_status(job_id, error_msg, status="failed", progress=100)
            raise FileNotFoundError(error_msg)
        
        # Load frames
        update_status(job_id, "Loading frames...", progress=20)
        first_frame = Image.open(first_frame_path).convert("RGB")
        last_frame = Image.open(last_frame_path).convert("RGB")
        
        # Process prompt if extension is enabled
        prompt = request.prompt
        if request.use_prompt_extend:
            update_status(job_id, "Extending prompt...", progress=25)
            prompt = extend_prompt(
                prompt=request.prompt,
                method=request.prompt_extend_method,
                model=request.prompt_extend_model,
                target_lang=request.prompt_extend_target_lang,
                seed=request.base_seed if request.base_seed >= 0 else random.randint(0, 2**32)
            )
        
        # Define max area based on dimensions
        height, width = first_frame.height, first_frame.width
        max_area = height * width
        for size_key, size_area in MAX_AREA_CONFIGS.items():
            if max_area > size_area:
                continue
            h, w = map(int, size_key.split('*'))
            if abs(h/w - height/width) < 0.1:  # Similar aspect ratio
                max_area = size_area
                break
                
        # Generate video
        update_status(job_id, "Generating video - this may take several minutes...", progress=30)
        
        # Set up generation parameters
        n_prompt = request.negative_prompt or config.sample_neg_prompt
        seed = request.base_seed if request.base_seed >= 0 else random.randint(0, 2**32)
        
        # Generate the video
        output_frames = flf2v_pipeline.generate(
            input_prompt=prompt,
            first_frame=first_frame,
            last_frame=last_frame,
            max_area=max_area,
            frame_num=request.frame_num,
            shift=request.sample_shift,
            sample_solver='unipc',  # Default value, not in request 
            sampling_steps=request.sample_steps,
            guide_scale=request.sample_guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=request.offload_model
        )
        
        # Check if generation was successful
        if output_frames is None or len(output_frames) == 0:
            error_msg = "Failed to generate video frames"
            logger.error(error_msg)
            update_status(job_id, error_msg, status="failed", progress=100)
            raise RuntimeError(error_msg)
        
        # Save output
        update_status(job_id, "Saving video...", progress=90)
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"flf2v_{timestamp}_{seed}.mp4"
        output_filepath = os.path.join(output_path, output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Save frames as video - use fps from request
        export_to_video(output_frames, output_filepath, request.fps)
        
        # Final update
        update_status(
            job_id, 
            "Video generation complete",
            status="complete",
            progress=100,
            video_preview=output_filepath
        )
        
        return {"status": "success", "video_path": output_filepath}
        
    except Exception as e:
        logger.error(f"Error generating first-last-frame video: {str(e)}")
        logger.error(traceback.format_exc())
        update_status(job_id, f"Error: {str(e)}", status="failed", progress=100)
        raise e


@torch.no_grad()
def process(request: WanJobSettings, device: Optional[str] = None):
    """Process a WAN job request"""
    # Ensure CUDA is available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for WAN processing")
    
    job_id = request.job_id
    
    # Check if this is a multi-process job (last frame part)
    is_last_frame_job = job_id.endswith("_lastframe")
    if is_last_frame_job:
        logger.info(f"Processing last frame job: {job_id}")
        # Extract the base job ID to link results properly
        base_job_id = job_id.replace("_lastframe", "")
        logger.info(f"Base job ID: {base_job_id}")
    
    # Initialize job status
    update_status(job_id, "Starting WAN processing...", "running", progress=0)
    
    try:
        # Handle segments if provided (for consistency with FramePack)
        if hasattr(request, 'segments') and request.segments:
            logger.info(f"Found {len(request.segments)} segments in the request")
            
            # Process segments - convert to proper format and fix paths
            segments = []
            for segment in request.segments:
                # Convert segment to dict if it's an object
                if hasattr(segment, 'model_dump'):
                    segment = segment.model_dump()
                
                # Fix image path - extract basename and join with upload_path
                image_path = segment.get('image_path', '')               
                # Handle data URLs - leave them as is
                
                segments.append(segment)
            
            # Set task based on number of segments
            if request.task is None or request.task == "":
                if len(segments) == 1:
                    request.task = "i2v-14B"
                    logger.info("Auto-selected task: i2v-14B (Image to Video)")
                else:
                    request.task = "flf2v-14B"
                    logger.info("Auto-selected task: flf2v-14B (First-Last Frame to Video)")
            
            # Extract image paths from segments based on task
            if request.task == "i2v-14B":
                # Use the first segment's image
                if segments:
                    image_path = segments[0].get('image_path')
                    if image_path:
                        logger.info(f"Using first segment image: {image_path}")
                        request.image = image_path
                    
            elif request.task == "flf2v-14B":
                # Use first and last segments for first-last frame
                if len(segments) >= 2:
                    first_image = segments[0].get('image_path')
                    last_image = segments[-1].get('image_path')
                    
                    if first_image:
                        logger.info(f"Using first segment image as first frame: {first_image}")
                        request.first_frame = first_image
                    
                    if last_image:
                        logger.info(f"Using last segment image as last frame: {last_image}")
                        request.last_frame = last_image
        
        logger.info(f"Request: {request}")
        
        # Process based on task
        if request.task == "i2v-14B":
            logger.info("Using Image to Video task (i2v-14B)")
            result = image_to_video(request)
        elif request.task == "flf2v-14B":
            logger.info("Using First-Last Frame to Video task (flf2v-14B)")
            # Determine sample_fps - use fps from request if it exists
            sample_fps = request.fps
            result = first_last_frame_to_video(request)
        else:
            raise ValueError(f"Invalid task type: {request.task}")
        
        # If this is a last frame job, update both jobs
        if is_last_frame_job:
            # Update the last frame job normally
            update_status(
                job_id,
                "WAN processing completed successfully",
                "completed",
                progress=100,
                video_preview=result
            )
            
            # Also update the base job
            update_status(
                base_job_id,
                "Last frame processing completed",
                "running",
                progress=None,
                video_preview=None
            )
        else:
            # Normal job update
            update_status(
                job_id,
                "WAN processing completed successfully",
                "completed",
                progress=100,
                video_preview=result
            )
        
        return {
            "status": "completed",
            "message": "WAN processing completed successfully",
            "result": result
        }
        
    except Exception as e:
        err_msg = f"WAN processing failed: {str(e)}"
        logger.error(err_msg)
        logger.exception(e)
        
        update_status(job_id, err_msg, "failed", progress=0)
        
        return {
            "status": "failed",
            "message": err_msg,
            "result": None
        } 