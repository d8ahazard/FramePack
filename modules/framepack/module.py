import sys
import os

# Add the modules path to the python path
sys.path.insert(0, os.path.dirname(__file__))
import logging
import shutil
import subprocess
import time
import traceback
from typing import Union

import einops
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, \
    SiglipVisionModel

from datatypes.datatypes import JobStatus, DynamicSwapInstaller
from handlers.job_queue import job_statuses, save_job_data
from handlers.model import check_download_model
from handlers.path import output_path, upload_path
from handlers.vram import fake_diffusers_current_device, get_cuda_free_memory_gb, \
    move_model_to_device_with_memory_preservation, unload_complete_models, load_model_as_complete, gpu, high_vram, \
    offload_model_from_device_for_memory_preservation
from modules.framepack.datatypes import FramePackJobSettings
from modules.framepack.diffusers_helper.bucket_tools import find_nearest_bucket
from modules.framepack.diffusers_helper.clip_vision import hf_clip_vision_encode
from modules.framepack.diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode_fake, vae_decode
from modules.framepack.diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from modules.framepack.diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from modules.framepack.diffusers_helper.utils import crop_or_pad_yield_mask, resize_and_center_crop, soft_append_bcthw, \
    save_bcthw_as_mp4

logger = logging.getLogger(__name__)
text_encoder = None
text_encoder_2 = None
image_encoder = None
vae = None
transformer = None
tokenizer = None
tokenizer_2 = None
feature_extractor = None
adaptive_memory_management = True

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
    logger.info('transformer.high_quality_fp32_output_for_inference = True')

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


def update_status(job_id, status, progress: int = None, latent_preview: str = None, video_preview: str = None):
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    job_status.status = status
    if latent_preview:
        job_status.current_latents = latent_preview
    if video_preview:
        job_status.result_video = video_preview
    if progress is not None:
        job_status.progress = progress

    # Persist updated job status
    save_job_data(job_id, job_status.to_dict())


@torch.no_grad()
def worker(job_id, input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg,
           gs, rs,
           gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution, segment_index=None, progress_callback=None):

    segment_name = f"{job_id}_segment_{segment_index + 1}" if segment_index is not None else job_id
    output_filename = os.path.join(output_path, f"{segment_name}.mp4")

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    global adaptive_memory_management, text_encoder, text_encoder_2, image_encoder, vae, transformer, tokenizer, tokenizer_2, feature_extractor
    adaptive_memory_management = enable_adaptive_memory
    load_models()

    if transformer is None or vae is None or image_encoder is None:
        raise ValueError("Transformer, VAE, or image encoder is not loaded properly")

    if not isinstance(transformer, HunyuanVideoTransformer3DModelPacked):
        raise ValueError("Transformer is not of type HunyuanVideoTransformer3DModelPacked")

    if not isinstance(vae, AutoencoderKLHunyuanVideo):
        raise ValueError("VAE is not of type AutoencoderKLHunyuanVideo")

    if not isinstance(image_encoder, SiglipVisionModel):
        raise ValueError("Image encoder is not of type SiglipVisionModel")


    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(image_encoder, vae, transformer)

        # Text encoding
        update_status(job_id, "Text encoding ...", 0)

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer,
                                                               tokenizer_2)

        # Unload text encoders
        load_model_as_complete(text_encoder, unload=True, target_device="cpu")
        load_model_as_complete(text_encoder_2, unload=True, target_device="cpu")

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image (start frame)
        update_status(job_id, "Processing start frame ...", 0)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(output_path, f'{job_id}_start.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # Processing end image (if provided)
        has_end_image = end_image is not None
        end_image_np = None
        end_image_pt = None
        end_latent = None

        if has_end_image:
            update_status(job_id, "Processing end frame ...", 0)

            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)

            Image.fromarray(end_image_np).save(os.path.join(output_path, f'{job_id}_end.png'))

            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        update_status(job_id, "VAE encoding ...", 0)

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)

        # CLIP Vision
        update_status(job_id, "CLIP Vision encoding ...", 0)

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings or use a weighted approach
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

        # Unload image encoder and feature extractor
        load_model_as_complete(image_encoder, unload=True, target_device="cpu")

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        update_status(job_id, "Start sampling ...", 0)

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # 将迭代器转换为列表
        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            # Check for job cancellation
            job_status = job_statuses.get(job_id)
            if job_status and job_status.status == "cancelled":
                logger.info(f"Job {job_id} was cancelled during processing")
                # This will cause the sampling to stop at the current step
                raise KeyboardInterrupt('User ends the task.')

            print(
                f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
                [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split(
                [1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Use end image latent for the first section if provided
            if has_end_image and is_first_section and end_latent is not None:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu,
                                                              preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                # Check if the job has been cancelled
                job_status = job_statuses.get(job_id)
                if job_status and job_status.status == "cancelled":
                    logger.info(f"Job {job_id} was cancelled during processing")
                    # This will cause the sampling to stop at the current step
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                update_status(job_id, desc, percentage, latent_preview=preview)
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
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
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            # See if we can just keep the VAE and transformer on GPU
            # if not high_vram:
            #     offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            #     load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # if not high_vram:
            #     unload_complete_models()

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
            update_status(job_id, desc, 100, None, output_filename)

            if is_last_section:
                break
    except:
        traceback.print_exc()

        # if not high_vram:
            # unload_complete_models(
            #     text_encoder, text_encoder_2, image_encoder, vae, transformer
            # )
    # Clean up start and end images, if they exist
    if os.path.exists(os.path.join(output_path, f'{job_id}_start.png')):
        os.remove(os.path.join(output_path, f'{job_id}_start.png'))
    if os.path.exists(os.path.join(output_path, f'{job_id}_end.png')):
        os.remove(os.path.join(output_path, f'{job_id}_end.png'))

    update_status(job_id, "Generation completed", 100)
    return output_filename


@torch.no_grad()
def worker_multi_segment(
        job_id,
        segments,
        global_prompt,
        negative_prompt,
        seed,
        steps,
        guidance_scale,
        rs=0.7,
        gpu_memory_preservation=6.0,
        use_teacache=True,
        mp4_crf=16,
        enable_adaptive_memory=True,
        resolution=640,
        latent_window_size=9,
        include_last_frame=False
):
    # Helper function to safely get value from segment (supporting both dict and object)
    def get_segment_value(segment, key, default=None):
        if segment is None:
            return default
        if hasattr(segment, key):
            return getattr(segment, key)
        elif isinstance(segment, dict) and key in segment:
            return segment[key]
        elif isinstance(segment, dict) and hasattr(segment, 'get'):
            return segment.get(key, default)
        return default

    # Import here to avoid circular imports
    load_models()
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    job_status.status = "running"
    job_status.message = "Starting multi-segment generation..."
    job_status.progress = 0
    job_statuses[job_id] = job_status

    # Save initial status to disk
    save_job_data(job_id, job_status.to_dict())

    if not segments:
        job_status.status = "failed"
        job_status.message = "No segments provided"
        save_job_data(job_id, job_status.to_dict())
        return None

    # Calculate total generation effort
    # Each segment takes steps * frames_per_segment amount of work
    total_steps = 0
    segment_workloads = []
    segment_dims = []
    for segment in segments:
        segment_duration = get_segment_value(segment, 'duration', 1.0)
        segment_frames = int(segment_duration * 30)
        segment_steps = steps
        segment_workload = segment_steps * segment_frames
        segment_workloads.append(segment_workload)
        total_steps += segment_workload
        segment_image = get_segment_value(segment, 'image_path')
        # Open the image and get the dimensions
        image = Image.open(segment_image)
        bucket = find_nearest_bucket(image.width, image.height, resolution)
        segment_dims.append((image.width, image.height, bucket))

    # Find the bucket dim for each segment, get a common one, and if they don't all match, resize and replace the path
    # Get the most common bucket dimensions
    bucket_counts = {}
    for _, _, bucket in segment_dims:
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
        else:
            bucket_counts[bucket] = 1
    
    # Find the most common bucket
    common_bucket = max(bucket_counts.items(), key=lambda x: x[1])[0]
    
    # Resize images if needed
    for i, segment in enumerate(segments):
        w, h, bucket = segment_dims[i]
        if bucket != common_bucket:
            # Need to resize this image
            segment_image = get_segment_value(segment, 'image_path')
            image = Image.open(segment_image)
            
            # Resize to the common bucket
            common_width, common_height = common_bucket
            resized_image = resize_and_center_crop(np.array(image), common_width, common_height)
            
            # Save the resized image with a new name
            resized_path = segment_image.replace('.', f'_resized_{common_width}x{common_height}.')
            Image.fromarray(resized_image).save(resized_path)
            
            # Update the segment path
            if hasattr(segment, 'image_path'):
                segment.image_path = resized_path
            elif isinstance(segment, dict):
                segment['image_path'] = resized_path

    # Set up progress tracking variables
    completed_steps = 0

    master_job_id = job_id
    master_temp = os.path.join(output_path, f"{master_job_id}_temp")
    os.makedirs(master_temp, exist_ok=True)

    segment_paths = []

    # Check if the job has been cancelled
    if job_status.status == "cancelled":
        logger.info(f"Job {job_id} was cancelled before processing started")
        try:
            shutil.rmtree(master_temp)
        except Exception as e:
            logger.info(f"Cleanup error: {e}")
        return None

    # Single segment case - simple animation from one image
    if len(segments) == 1:
        job_status.message = "Processing single image video..."
        save_job_data(job_id, job_status.to_dict())

        current_segment = segments[0]

        try:
            # Check if the job has been cancelled
            if job_status.status == "cancelled":
                logger.info(f"Job {job_id} was cancelled before single segment processing started")
                try:
                    shutil.rmtree(master_temp)
                except Exception as e:
                    logger.info(f"Cleanup error: {e}")
                return None

            # Get the image path directly from the segment
            image_path = get_segment_value(current_segment, 'image_path')

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load start image
            logger.info(f"Loading single image from: {image_path}")
            start_image = np.array(Image.open(image_path).convert('RGB'))

            # Use segment prompt if provided, concatenate with global prompt
            segment_prompt = get_segment_value(current_segment, 'prompt', '').strip()
            if segment_prompt:
                # Concatenate with global prompt
                combined_prompt = f"{global_prompt.strip()}, {segment_prompt}"
            else:
                combined_prompt = global_prompt

            # Get segment duration (longer for single image)
            segment_duration = get_segment_value(current_segment, 'duration', 3.0)

            # Custom callback to update master progress
            def progress_callback(segment_progress):
                overall_progress = int((segment_progress / 100) * 100)
                job_status.progress = overall_progress

                # Update job status in filesystem
                save_job_data(job_id, job_status.to_dict())

                # Directly update UI via websocket for more immediate feedback
                try:
                    from handlers.socket import update_status
                    update_status(
                        job_id=job_id,
                        status=job_status.status,
                        progress=overall_progress,
                        message=job_status.message
                    )
                except Exception as e:
                    logger.info(f"Error updating status via websocket: {e}")

            # Generate the video with the single image
            segment_output = worker(
                job_id=master_job_id,
                input_image=start_image,
                end_image=None,  # No end image for single image case
                prompt=combined_prompt,
                n_prompt=negative_prompt,
                seed=seed,
                total_second_length=segment_duration,
                latent_window_size=latent_window_size,
                steps=steps,
                cfg=guidance_scale,
                gs=guidance_scale,
                rs=rs,
                gpu_memory_preservation=gpu_memory_preservation,
                use_teacache=use_teacache,
                mp4_crf=mp4_crf,
                enable_adaptive_memory=enable_adaptive_memory,
                resolution=resolution,
                segment_index=0,
                progress_callback=progress_callback
            )

            if segment_output:
                job_status.status = "completed"
                job_status.progress = 100
                job_status.message = "Single image video generation completed!"
                job_status.result_video = segment_output
                save_job_data(job_id, job_status.to_dict())
                return segment_output
            else:
                job_status.status = "failed"
                job_status.message = "Failed to generate video from single image"
                save_job_data(job_id, job_status.to_dict())
                return None

        except Exception as e:
            error_msg = f"Error processing single image: {str(e)}"
            logger.info(error_msg)
            job_status.status = "failed"
            job_status.message = error_msg
            save_job_data(job_id, job_status.to_dict())
            return None

    job_status.message = "Processing multi-segment video..."
    save_job_data(job_id, job_status.to_dict())
    num_segments = len(segments)
    last_segment = segments[-1]
    use_last_frame = get_segment_value(last_segment, 'use_last_frame', False)
    logger.info(f"use_last_frame: {use_last_frame}")
    # framepack_settings = job_status.job_settings.get('framepack', {})
    # use_last_frame = framepack_settings.get('include_last_frame',
    #                                         False) if job_status.job_settings else False

    segment_pairs = []
    if len(segments) > 1:
        for i in range(num_segments - 1):
            segment_pairs.append((segments[i], segments[i + 1]))
        if use_last_frame:
            segment_pairs.append((segments[-1], None))
    else:
        segment_pairs.append((segments[0], None))

    seg_no = 0

    logger.info(f"segment_pairs: {segment_pairs}")

    for (start_segment, end_segment) in segment_pairs:
        logger.info(f"Processing segments: {start_segment} and {end_segment}")
        # Check if the job has been cancelled
        job_status = job_statuses.get(job_id)
        if job_status and job_status.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled before segment {seg_no} processing started")
            try:
                shutil.rmtree(master_temp)
            except Exception as e:
                logger.info(f"Cleanup error: {e}")
            return None

        # Calculate progress up to this segment
        previous_segments_progress = 0
        for j in range(seg_no + 1, num_segments):
            previous_idx = num_segments - 1 - j
            previous_segments_progress += segment_workloads[previous_idx]

        # Use individual segment prompts if provided, concatenate with global prompt
        segment_prompt = get_segment_value(start_segment, 'prompt', '').strip()
        if segment_prompt:
            # Concatenate with global prompt
            combined_prompt = f"{global_prompt.strip()}, {segment_prompt}"
        else:
            combined_prompt = global_prompt

        # Get segment duration
        segment_duration = get_segment_value(start_segment, 'duration', 3.0)

        try:
            # Get the image path directly from the segment
            image_path = get_segment_value(start_segment, 'image_path')

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load start image
            logger.info(f"Loading start image from: {image_path}")
            start_image = np.array(Image.open(image_path).convert('RGB'))

            # Load end image if not the last segment
            end_image = None
            if end_segment is not None:
                end_image_path = get_segment_value(end_segment, 'image_path')
                if end_image_path is None:
                    raise ValueError("End image path is missing in the segment data")

                if not os.path.exists(end_image_path):
                    raise FileNotFoundError(f"End image file not found: {end_image_path}")

                logger.info(f"Loading end image from: {end_image_path}")
                end_image = np.array(Image.open(end_image_path).convert('RGB'))


        except Exception as e:
            error_msg = f"Error loading image for segment {seg_no}: {str(e)}"
            logger.info(error_msg)
            job_status.status = "failed"
            job_status.message = error_msg
            save_job_data(job_id, job_status.to_dict())
            return None

        job_status.message = f"Processing segment {seg_no}/{num_segments}..."

        # Custom callback to update master progress based on segment's contribution to overall workload
        def progress_callback(segment_progress):
            segment_contribution = segment_workloads[seg_no]
            segment_completed = (segment_progress / 100) * segment_contribution
            overall_completed = previous_segments_progress + segment_completed
            overall_progress = int((overall_completed / total_steps) * 100)
            job_status.progress = overall_progress

            # Update job status in filesystem
            save_job_data(job_id, job_status.to_dict())

            # Directly update UI via websocket for more immediate feedback
            # try:
            #     queue_broadcast(
            #         job_id=job_id,
            #         status=job_status.status,
            #         progress=overall_progress,
            #         message=job_status.message
            #     )
            # except Exception as e:
            #     logger.info(f"Error updating status via websocket: {e}")

        # Clear cache if needed
        curr_free = get_cuda_free_memory_gb(gpu)
        if not high_vram and curr_free < 2.0:
            torch.cuda.empty_cache()
            if curr_free < 1.0:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

        logger.info(f"Generating segment {seg_no} with length {segment_duration} seconds...")
        # Generate this segment
        segment_output = worker(
            job_id=master_job_id,
            input_image=start_image,
            end_image=end_image,
            prompt=combined_prompt,
            n_prompt=negative_prompt,
            seed=seed,
            total_second_length=segment_duration,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=guidance_scale,
            gs=guidance_scale,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            enable_adaptive_memory=enable_adaptive_memory,
            resolution=resolution,
            segment_index=seg_no,
            progress_callback=progress_callback
        )

        if segment_output:
            logger.info(f"Segment {seg_no} generated: {segment_output}")
            segment_paths.append((seg_no, segment_output))
            completed_steps += segment_workloads[seg_no]
        else:
            job_status.status = "failed"
            job_status.message = f"Failed to generate segment {seg_no}"
            save_job_data(job_id, job_status.to_dict())
            return None

        # Increment segment number
        seg_no += 1

    # Concatenate all segments in order
    concat_txt = os.path.join(master_temp, "concat_list.txt")
    logger.info(f"Concatenating segments in order: {concat_txt}")
    cat_txt = ""
    with open(concat_txt, 'w') as f:
        for seg_no, path in sorted(segment_paths, key=lambda x: x[0]):
            # Ensure an absolute path with proper escaping for FFmpeg
            abs_path = os.path.abspath(path).replace('\\', '/')
            cat_txt += f"file '{abs_path}'\n"
            f.write(f"file '{abs_path}'\n")
    logger.info(f"cat_txt: {cat_txt}")
    final_output = os.path.join(output_path, f"{master_job_id}_final.mp4")
    logger.info(f"Final output: {final_output}")
    try:
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_txt, '-c', 'copy', final_output
        ], check=True)
    except Exception as e:
        job_status.status = "failed"
        job_status.message = f"Failed to concatenate segments: {str(e)}"
        save_job_data(job_id, job_status.to_dict())
        return None

    # Clean up temporary files
    if os.path.exists(concat_txt):
        try:
            os.remove(concat_txt)
        except Exception as e:
            logger.info(f"Error removing concat file {concat_txt}: {e}")

    if os.path.exists(final_output):
        for seg_no, path in segment_paths:
            try:
                os.remove(path)
                logger.info(f"Removed segment file {path}")
            except Exception as e:
                logger.info(f"Error removing segment file {path}: {e}")

    if not high_vram:
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae, transformer
        )
    try:
        shutil.rmtree(master_temp)
    except Exception as e:
        logger.info(f"Cleanup error: {e}")

    job_status.status = "completed"
    job_status.progress = 100
    job_status.message = "Video generation completed!"
    job_status.result_video = final_output
    save_job_data(job_id, job_status.to_dict())

    return final_output


@torch.no_grad()
def process(request: FramePackJobSettings):
    request_dict = request.model_dump()

    # Convert SegmentConfig objects to dictionaries
    if 'segments' in request_dict and request_dict['segments']:
        segments = request_dict['segments']
        segments = [
            segment.model_dump() if hasattr(segment, 'model_dump') else segment
            for segment in segments
        ]
        # Replace "/uploads/" with "upload_path" in segments
        for segment in segments:
            segment['image_path'] = segment['image_path'].replace("/uploads/", "")
            # This shouldn't happen, but just in case, handler "\uploads\" to
            segment['image_path'] = segment['image_path'].replace("\\uploads\\", "")
            segment['image_path'] = os.path.join(upload_path, segment['image_path'])
        request_dict['segments'] = segments
    return worker_multi_segment(**request_dict)
