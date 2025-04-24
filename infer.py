import os
import re
import PIL
import subprocess
import shutil
import traceback
import argparse
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union

import torch
import einops
import numpy as np
from huggingface_hub import snapshot_download
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel
)
from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models,
    load_model_as_complete
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# ----------------------------------------
# Model classes and helpers
# ----------------------------------------

class JobStatus:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.message = "Initializing..."
        self.preview_image = None
        self.result_video = None
        self.segments = []

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result_video": self.result_video,
            "segments": self.segments
        }

class SegmentConfig(BaseModel):
    image_path: str
    prompt: str
    duration: float

class VideoRequest(BaseModel):
    global_prompt: str
    negative_prompt: str
    segments: List[SegmentConfig]
    seed: int = 31337
    steps: int = 25
    guidance_scale: float = 10.0
    use_teacache: bool = True
    enable_adaptive_memory: bool = True
    resolution: int = 640
    mp4_crf: int = 16
    gpu_memory_preservation: float = 6.0

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result_video: Optional[str] = None
    segments: List[str] = []

class ErrorResponse(BaseModel):
    error: str

class UploadResponse(BaseModel):
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None

class GenerateResponse(BaseModel):
    job_id: str

# Global job tracking
job_statuses = {}
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)
    
# ----------------------------------------
# Model download & loading
# ----------------------------------------

def check_download_model(repo_id, subfolder=None, retries=3, use_auth_token=None):
    """
    Downloads a model from Hugging Face Hub to a local models directory.
    
    Args:
        repo_id: The Hugging Face model repository ID
        subfolder: Optional subfolder within the model repository
        retries: Number of download retries on failure
        use_auth_token: Optional auth token for private models
        
    Returns:
        str: Path to the downloaded model for use with from_pretrained
    """
    # Define the models directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a sanitized directory name for this specific model
    model_name = repo_id.replace('/', '_')
    model_dir = os.path.join(models_dir, model_name)
    
    # Check if the model is already downloaded
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Using cached model {repo_id} from {model_dir}")
    else:
        print(f"Downloading {repo_id} to {model_dir}...")
        
        for attempt in range(retries):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_files_only=True,
                    token=use_auth_token,
                    max_workers=4  # Limit concurrent downloads
                )
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Download attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"All {retries} download attempts failed for {repo_id}")
                    raise RuntimeError(f"Failed to download model {repo_id}: {e}")
    
    # Verify the model directory has content
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        raise RuntimeError(f"Model directory {model_dir} is empty after download")
    
    # Return the appropriate path
    if subfolder:
        subfolder_path = os.path.join(model_dir, subfolder)
        if not os.path.exists(subfolder_path):
            raise ValueError(f"Subfolder '{subfolder}' doesn't exist in {model_dir}")
        return subfolder_path
    
    return model_dir


def preload_all_models(use_auth_token=None):
    """
    Preloads all required models to the local cache.
    
    Args:
        use_auth_token: Optional auth token for private models
        
    Returns:
        dict: Paths to all downloaded models
    """
    print("Preloading all required models...")
    
    model_repos = {
        "hunyuan": "hunyuanvideo-community/HunyuanVideo",
        "flux": "lllyasviel/flux_redux_bfl",
        "framepack": "lllyasviel/FramePackI2V_HY"
    }
    
    model_paths = {}
    
    for name, repo_id in model_repos.items():
        try:
            path = check_download_model(repo_id, use_auth_token=use_auth_token)
            model_paths[name] = path
            print(f"Successfully preloaded {name} model from {repo_id}")
        except Exception as e:
            print(f"Error preloading {name} model: {e}")
            raise
    
    print("All models preloaded successfully!")
    return model_paths


# ----------------------------------------
# VRAM & model-loading setup
# ----------------------------------------
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 32
adaptive_memory_management = True

print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')
print(f'Adaptive Memory Management: {adaptive_memory_management}')

text_encoder = None
text_encoder_2 = None
image_encoder = None
vae = None
transformer = None
tokenizer = None
tokenizer_2 = None
feature_extractor = None

models_loaded = False

# Download and get paths for model repositories
hunyuan_path = check_download_model("hunyuanvideo-community/HunyuanVideo")
flux_path = check_download_model("lllyasviel/flux_redux_bfl")
framepack_path = check_download_model("lllyasviel/FramePackI2V_HY")

def load_models():
    global text_encoder, text_encoder_2, image_encoder, vae, transformer, tokenizer, tokenizer_2, feature_extractor, models_loaded
    if models_loaded:
        return
    # Load models from local paths with appropriate subfolders
    text_encoder = LlamaModel.from_pretrained(
        os.path.join(hunyuan_path, 'text_encoder'),
        torch_dtype=torch.float16
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        os.path.join(hunyuan_path, 'text_encoder_2'),
        torch_dtype=torch.float16
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(
        os.path.join(hunyuan_path, 'tokenizer')
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        os.path.join(hunyuan_path, 'tokenizer_2')
    )
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        os.path.join(hunyuan_path, 'vae'),
        torch_dtype=torch.float16
    ).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained(
        os.path.join(flux_path, 'feature_extractor')
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        os.path.join(flux_path, 'image_encoder'),
        torch_dtype=torch.float16
    ).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        framepack_path,
        torch_dtype=torch.bfloat16
    ).cpu()

    # eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # VAE slicing/tiling
    vae.enable_slicing()
    vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    # dtype cast
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    models_loaded = True

    # no grads
    for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        m.requires_grad_(False)

    # model offloading or full-load
    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        for m in (text_encoder, text_encoder_2, image_encoder, vae, transformer):
            m.to(gpu)


# ----------------------------------------
# Worker: single-segment generation
# ----------------------------------------
@torch.no_grad()
def worker(
    job_id,
    input_image,
    end_image,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    enable_adaptive_memory,
    resolution,
    segment_index=None,
    master_job_id=None
):
    global adaptive_memory_management, job_statuses
    adaptive_memory_management = enable_adaptive_memory
    
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    
    # how many latent sections
    total_latent_sections = int(
        max(round((total_second_length * 30) / (latent_window_size * 4)), 1)
    )

    # decide job_id + base_name
    if segment_index is not None and master_job_id:
        job_id = master_job_id
        segment_number = segment_index + 1
    else:
        segment_number = 1

    base_name = f"{job_id}_segment_{segment_number}"
    output_path = os.path.join(outputs_folder, f"{base_name}.mp4")
    temp_dir = os.path.join(outputs_folder, f"{base_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    job_status.message = "Starting..."
    job_status.progress = 0
    
    try:
        # -------------------------
        # Memory management tiers
        # -------------------------
        models_to_keep_in_memory = []
        if high_vram or adaptive_memory_management:
            if free_mem_gb >= 26:
                image_encoder.to(gpu)
                vae.to(gpu)
                transformer.to(gpu)
                models_to_keep_in_memory = [
                    image_encoder,
                    vae,
                    transformer
                ]
                print("Ultra high memory mode: Keeping all models")
            elif free_mem_gb >= 23:
                transformer.to(gpu)
                models_to_keep_in_memory = [
                    transformer,
                    vae
                ]
                print("High memory mode: Keeping transform & text encoders")
            elif free_mem_gb >= 4:
                if free_mem_gb >= 6:
                    image_encoder.to(gpu)
                    vae.to(gpu)
                    print("Medium memory mode: + image encoder & VAE")
                else:
                    print("Medium memory mode: text encoders only")
            else:
                print("Low memory mode: no preloads")
        else:
            print("Compatibility mode: unloading all")
            unload_complete_models(
                text_encoder,
                text_encoder_2,
                image_encoder,
                vae,
                transformer
            )

        # -------------------------
        # Text encoding
        # -------------------------
        job_status.message = "Text encoding..."
        job_status.progress = 5

        # We literally only need the TENC once, so just load and unload when done
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2
        )
        if cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2
            )

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Hot nasty speed
        unload_complete_models(text_encoder, text_encoder_2)

        # -------------------------
        # Image processing
        # -------------------------
        job_status.message = "Image processing..."
        job_status.progress = 10

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_np = resize_and_center_crop(input_image, width, height)
        Image.fromarray(input_np).save(os.path.join(outputs_folder, f"{job_id}.png"))
        input_pt = (
            torch.from_numpy(input_np).float() / 127.5 - 1
        ).permute(2, 0, 1)[None, :, None]

        has_end_image = end_image is not None
        if has_end_image:
            job_status.message = "Processing end frame..."
            job_status.progress = 15
            
            end_np = resize_and_center_crop(end_image, width, height)
            Image.fromarray(end_np).save(os.path.join(outputs_folder, f"{job_id}_end.png"))
            end_pt = (
                torch.from_numpy(end_np).float() / 127.5 - 1
            ).permute(2, 0, 1)[None, :, None]

        # -------------------------
        # VAE encoding
        # -------------------------
        job_status.message = "VAE encoding..."
        job_status.progress = 20

        if vae not in models_to_keep_in_memory:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_pt, vae)
        if has_end_image:
            end_latent = vae_encode(end_pt, vae)

        # -------------------------
        # CLIP Vision encoding
        # -------------------------
        job_status.message = "CLIP Vision encoding..."
        job_status.progress = 25
        
        load_model_as_complete(image_encoder, target_device=gpu)

        vis_out = hf_clip_vision_encode(input_np, feature_extractor, image_encoder)
        image_emb = vis_out.last_hidden_state
        if has_end_image:
            end_vis = hf_clip_vision_encode(end_np, feature_extractor, image_encoder)
            image_emb = (image_emb + end_vis.last_hidden_state) / 2

        # Unload image_encoder if not high VRAM
        if not high_vram:
            unload_complete_models(image_encoder)

        # cast dtypes
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_emb = image_emb.to(transformer.dtype)

        # -------------------------
        # Sampling
        # -------------------------
        job_status.message = "Start sampling..."
        job_status.progress = 30

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        history_latents = torch.zeros(
            (1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated = 0

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # ensure transformer on GPU if needed
        if transformer not in models_to_keep_in_memory:
            preserved = gpu_memory_preservation
            if use_teacache:
                preserved = max(preserved, 8)
            move_model_to_device_with_memory_preservation(
                transformer,
                target_device=gpu,
                preserved_memory_gb=preserved
            )

        # init teaCache
        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=min(steps, 20))
        else:
            transformer.initialize_teacache(enable_teacache=False)

        for pad in latent_paddings:
            is_last = (pad == 0)
            is_first = (pad == latent_paddings[0])
            pad_size = pad * latent_window_size

            # build indices
            total_len = 1 + pad_size + latent_window_size + 1 + 2 + 16
            indices = torch.arange(0, total_len).unsqueeze(0)
            (clean_pre,
             blank,
             latent_indices,
             clean_post,
             clean2x_idx,
             clean4x_idx) = indices.split(
                [1, pad_size, latent_window_size, 1, 2, 16], dim=1
            )
            clean_indices = torch.cat([clean_pre, clean_post], dim=1)

            # build clean_latents
            clean_pre_latents = start_latent.to(history_latents)
            pre, two, sixteen = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_pre_latents, pre], dim=2)

            if has_end_image and is_first:
                clean_post_latents = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_pre_latents, clean_post_latents], dim=2)

            # re-init teaCache each loop
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = vae_decode_fake(d['denoised'])
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                
                # Save preview as temp file
                preview_path = os.path.join(temp_dir, "preview.jpg")
                Image.fromarray(preview).save(preview_path)
                
                step = d['i'] + 1
                pct = int(100.0 * step / steps)
                base_progress = 30
                step_progress = int(70 * (pad / len(latent_paddings)) + (70 / len(latent_paddings)) * (step / steps))
                total_progress = base_progress + step_progress
                
                job_status.progress = total_progress
                job_status.message = f"Generated {max(0, (total_generated * 4 - 3))} frames, {(max(0, (total_generated * 4 - 3)) / 30):.2f}s so far"
                job_status.preview_image = preview_path
                
                return

            generated = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_emb,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_indices,
                clean_latents_2x=two,
                clean_latent_2x_indices=clean2x_idx,
                clean_latents_4x=sixteen,
                clean_latent_4x_indices=clean4x_idx,
                callback=callback
            )

            if is_last:
                generated = torch.cat([start_latent.to(generated), generated], dim=2)

            total_generated += generated.shape[2]
            history_latents = torch.cat([generated.to(history_latents), history_latents], dim=2)

            load_model_as_complete(vae, target_device=gpu)

            # decode frames
            real = history_latents[:, :, :total_generated]
            if history_pixels is None:
                history_pixels = vae_decode(real, vae).cpu()
            else:
                frames_count = latent_window_size * 2 + (1 if is_last else 0)
                overlap = latent_window_size * 4 - 3
                curr_pix = vae_decode(real[:, :, :frames_count], vae).cpu()
                history_pixels = soft_append_bcthw(curr_pix, history_pixels, overlap)

            # overwrite segment file
            save_bcthw_as_mp4(history_pixels, output_path, fps=30, crf=mp4_crf)
            # Unload VAE if not high VRAM
            if not high_vram:
                unload_complete_models(vae)
                
            job_status.segments.append(output_path)
            
            if is_last:
                break

        job_status.status = "completed"
        job_status.progress = 100
        job_status.message = "Generation completed"
        job_status.result_video = output_path
        
        return output_path

    except Exception as e:
        job_status.status = "failed"
        job_status.message = str(e)
        traceback.print_exc()
        return None
    finally:
        if not high_vram and segment_index is None:
            unload_complete_models(
                text_encoder,
                text_encoder_2,
                image_encoder,
                vae,
                transformer
            )
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")


# ----------------------------------------
# Worker: multi-segment generation
# ----------------------------------------
@torch.no_grad()
def worker_multi_segment(
    job_id,
    segments,
    global_prompt,
    n_prompt,
    seed,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    enable_adaptive_memory,
    resolution,
    latent_window_size=9
):
    global job_statuses
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    job_status.message = "Starting multi-segment generation..."
    job_status.progress = 0
    job_statuses[job_id] = job_status
    
    if not segments:
        job_status.status = "failed"
        job_status.message = "No segments provided"
        return None
    
    master_job_id = job_id
    master_temp = os.path.join(outputs_folder, f"{master_job_id}_temp")
    os.makedirs(master_temp, exist_ok=True)

    segment_paths = []

    # Process segments in reverse order
    num_segments = len(segments)
    for i in range(num_segments - 1, -1, -1):
        seg_no = num_segments - i
        current_segment = segments[i]
        
        # Use individual segment prompts if provided, concatenate with global prompt
        segment_prompt = current_segment.get('prompt', '').strip()
        if segment_prompt:
            # Concatenate with global prompt
            combined_prompt = f"{global_prompt.strip()}, {segment_prompt}"
        else:
            combined_prompt = global_prompt
            
        # Get segment duration
        segment_duration = current_segment.get('duration', 3.0)
        
        # Load start and end images
        start_image = np.array(Image.open(current_segment['image_path']).convert('RGB'))
        end_image = None
        if i < num_segments - 1:
            end_image = np.array(Image.open(segments[i+1]['image_path']).convert('RGB'))
        
        job_status.message = f"Processing segment {seg_no}/{num_segments}..."
        job_status.progress = int((i / num_segments) * 100)
        
        # Clear cache if needed
        curr_free = get_cuda_free_memory_gb(gpu)
        if not high_vram and curr_free < 2.0:
            torch.cuda.empty_cache()
            if curr_free < 1.0:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

        # Generate this segment
        segment_output = worker(
            job_id=master_job_id,
            input_image=start_image,
            end_image=end_image,
            prompt=combined_prompt,
            n_prompt=n_prompt,
            seed=seed,
            total_second_length=segment_duration,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            enable_adaptive_memory=enable_adaptive_memory,
            resolution=resolution,
            segment_index=i,
            master_job_id=master_job_id
        )
        
        if segment_output:
            segment_paths.append((seg_no, segment_output))
        else:
            job_status.status = "failed" 
            job_status.message = f"Failed to generate segment {seg_no}"
            return None

    # Concatenate all segments in order
    concat_txt = os.path.join(master_temp, "concat_list.txt")
    with open(concat_txt, 'w') as f:
        for seg_no, path in sorted(segment_paths, key=lambda x: x[0]):
            # Ensure absolute path with proper escaping for FFmpeg
            abs_path = os.path.abspath(path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    final_output = os.path.join(outputs_folder, f"{master_job_id}.mp4")
    try:
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_txt, '-c', 'copy', final_output
        ], check=True)
    except Exception as e:
        job_status.status = "failed"
        job_status.message = f"Failed to concatenate segments: {str(e)}"
        return None

    if not high_vram:
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae, transformer
        )
    try:
        shutil.rmtree(master_temp)
    except Exception as e:
        print(f"Cleanup error: {e}")

    job_status.status = "completed"
    job_status.progress = 100
    job_status.message = "Video generation completed!"
    job_status.result_video = final_output
    
    return final_output

# ----------------------------------------
# FastAPI app setup
# ----------------------------------------
app = FastAPI(
    title="FramePack API",
    description="API for image to video generation using FramePack",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure logging to suppress job status logs
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.getMessage().find("GET /api/job_status/") >= 0)

# Apply the filter to the uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a middleware to include response headers
@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Upload folder for images
upload_folder = './uploads/'
os.makedirs(upload_folder, exist_ok=True)

# Create folders for static files if they don't exist
os.makedirs("static/images", exist_ok=True)

# ----------------------------------------
# API Routes
# ----------------------------------------

@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon"""
    return FileResponse("static/images/favicon.ico")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/api")
async def api_docs_redirect():
    """Redirect to API documentation"""
    return RedirectResponse(url="/api/docs")

@app.post("/api/upload_image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file to the server
    
    Returns:
        success: Whether the upload was successful
        filename: The filename on the server
        path: The path to the file on the server
    """
    # Get file content hash to check for duplicates
    file_content = await file.read()
    file_hash = hash(file_content)
    
    # Return file pointer to start for later use
    await file.seek(0)
    
    # Check if we already have this file or one with the same name
    original_filename = file.filename
    base_name, ext = os.path.splitext(original_filename)
    
    # First check if exact same file (by hash) exists
    for existing_file in os.listdir(upload_folder):
        existing_path = os.path.join(upload_folder, existing_file)
        if os.path.isfile(existing_path):
            with open(existing_path, "rb") as f:
                existing_content = f.read()
                if hash(existing_content) == file_hash:
                    # Same file content - reuse it
                    return UploadResponse(
                        success=True,
                        filename=existing_file,
                        path=existing_path
                    )
    
    # Generate a new unique filename
    timestamp = generate_timestamp()
    filename = f"{timestamp}_{original_filename}"
    file_path = os.path.join(upload_folder, filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return UploadResponse(
        success=True,
        filename=filename,
        path=file_path
    )

@app.post("/api/generate_video", response_model=GenerateResponse)
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
    job_status = JobStatus(job_id)
    job_statuses[job_id] = job_status
    
    # Process segments
    segments = []
    for segment in request_data.segments:
        segments.append({
            "image_path": segment.image_path,
            "prompt": segment.prompt,
            "duration": segment.duration
        })
    
    # Start generation in background
    background_tasks.add_task(
        worker_multi_segment,
        job_id=job_id,
        segments=segments,
        global_prompt=request_data.global_prompt,
        n_prompt=request_data.negative_prompt,
        seed=request_data.seed,
        steps=request_data.steps,
        cfg=1.0,  # Fixed parameter
        gs=request_data.guidance_scale,
        rs=0.0,   # Fixed parameter
        gpu_memory_preservation=request_data.gpu_memory_preservation,
        use_teacache=request_data.use_teacache,
        mp4_crf=request_data.mp4_crf,
        enable_adaptive_memory=request_data.enable_adaptive_memory,
        resolution=request_data.resolution
    )
    
    return GenerateResponse(job_id=job_id)

@app.get("/api/job_status/{job_id}", response_model=Union[JobStatusResponse, ErrorResponse])
async def get_job_status(job_id: str):
    """
    Get the status of a video generation job
    
    Args:
        job_id: The unique job identifier
    
    Returns:
        The job status information
    """
    if job_id not in job_statuses:
        return ErrorResponse(error="Job not found")
    
    status = job_statuses[job_id]
    return JobStatusResponse(**status.to_dict())

@app.get("/api/result_video/{job_id}")
async def get_result_video(job_id: str):
    """
    Get the video file for a completed job
    
    Args:
        job_id: The unique job identifier
    
    Returns:
        The video file
    """
    if job_id not in job_statuses or not job_statuses[job_id].result_video:
        return ErrorResponse(error="Video not found")
    
    video_path = job_statuses[job_id].result_video
    return FileResponse(
        path=video_path, 
        filename=f"{job_id}.mp4", 
        media_type="video/mp4"
    )

@app.get("/api/list_jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """
    List all active jobs
    
    Returns:
        A list of all active job statuses
    """
    return [JobStatusResponse(**status.to_dict()) for status in job_statuses.values()]

@app.get("/api/list_outputs")
async def list_outputs():
    """
    List all generated output videos
    
    Returns:
        A list of all output videos with metadata
    """
    outputs = []
    try:
        for filename in os.listdir(outputs_folder):
            if filename.endswith('.mp4'):
                file_path = os.path.join(outputs_folder, filename)
                # Skip temp directories and check if it's a file
                if not os.path.isfile(file_path) or '_temp' in filename:
                    continue
                    
                # Get file stats
                stats = os.stat(file_path)
                timestamp = stats.st_mtime
                
                # Create output entry
                output = {
                    "name": filename,
                    "path": f"/outputs/{filename}",
                    "timestamp": timestamp,
                    "size": stats.st_size
                }
                outputs.append(output)
        
        # Sort by timestamp (newest first)
        outputs.sort(key=lambda x: x["timestamp"], reverse=True)
        
    except Exception as e:
        print(f"Error listing outputs: {e}")
        import traceback
        traceback.print_exc()
    
    return outputs

@app.delete("/api/job/{job_id}", response_model=Union[dict, ErrorResponse])
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    
    Args:
        job_id: The unique job identifier
    
    Returns:
        Success message or error
    """
    if job_id not in job_statuses:
        return ErrorResponse(error="Job not found")
    
    status = job_statuses[job_id]
    
    # Delete result video if it exists
    if status.result_video and os.path.exists(status.result_video):
        try:
            os.remove(status.result_video)
        except Exception as e:
            print(f"Failed to delete video file: {e}")
    
    # Delete segment videos if they exist
    for segment in status.segments:
        if os.path.exists(segment):
            try:
                os.remove(segment)
            except Exception as e:
                print(f"Failed to delete segment file: {e}")
    
    # Remove job from statuses
    del job_statuses[job_id]
    
    return {"success": True, "message": f"Job {job_id} deleted"}

# Run the application
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true", help="Preload all models at startup")
    parser.add_argument("--hf_token", type=str, help="Hugging Face authentication token")
    args = parser.parse_args()
    
    # Preload models if requested
    if args.preload:
        model_paths = preload_all_models(use_auth_token=args.hf_token)
    
    # Load models
    load_models()
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port) 