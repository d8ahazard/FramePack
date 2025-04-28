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
    move_model_to_device_with_memory_preservation, unload_complete_models, load_model_as_complete, gpu, high_vram
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
        master_job_id=None,
        progress_callback=None
) -> Union[None, str]:
    # Import here to avoid circular imports

    global adaptive_memory_management, text_encoder, text_encoder_2, image_encoder, vae, transformer, tokenizer, tokenizer_2, feature_extractor
    adaptive_memory_management = enable_adaptive_memory
    load_models()
    job_status = job_statuses.get(job_id, JobStatus(job_id))
    job_status.status = "running"

    # Persist job status to disk
    save_job_data(job_id, job_status.to_dict())

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
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_name = f"{job_id}_segment_{segment_number}_{timestamp}"
    logger.info(f"base_name: {base_name}")
    output_file = os.path.join(output_path, f"{base_name}.mp4")
    
    job_status.message = "Starting..."
    job_status.progress = 0

    # Persist updated job status
    save_job_data(job_id, job_status.to_dict())
    if not isinstance(transformer, HunyuanVideoTransformer3DModelPacked):
        raise ValueError("Transformer is not of type HunyuanVideoTransformer3DModelPacked")

    if not isinstance(vae, AutoencoderKLHunyuanVideo):
        raise ValueError("VAE is not of type AutoencoderKLHunyuanVideo")

    if not isinstance(image_encoder, SiglipVisionModel):
        raise ValueError("Image encoder is not of type SiglipVisionModel")

    try:
        # -------------------------
        # Memory management tiers
        # -------------------------
        models_to_keep_in_memory = []
        free_mem_gb = get_cuda_free_memory_gb()
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
                logger.info("Ultra high memory mode: Keeping all models")
            elif free_mem_gb >= 23:
                transformer.to(gpu)
                models_to_keep_in_memory = [
                    transformer,
                    vae
                ]
                logger.info("High memory mode: Keeping transform & text encoders")
            elif free_mem_gb >= 4:
                if free_mem_gb >= 6:
                    image_encoder.to(gpu)
                    vae.to(gpu)
                    logger.info("Medium memory mode: + image encoder & VAE")
                else:
                    logger.info("Medium memory mode: text encoders only")
            else:
                logger.info("Low memory mode: no preloads")
        else:
            logger.info("Compatibility mode: unloading all")
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
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(5)

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
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(10)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_np = resize_and_center_crop(input_image, width, height)
        Image.fromarray(input_np).save(os.path.join(output_path, f"{job_id}.png"))
        input_pt = (
                           torch.from_numpy(input_np).float() / 127.5 - 1
                   ).permute(2, 0, 1)[None, :, None]

        has_end_image = end_image is not None
        end_pt = None
        end_np = None
        end_latent = None

        if has_end_image:
            job_status.message = "Processing end frame..."
            job_status.progress = 15
            save_job_data(job_id, job_status.to_dict())
            if progress_callback:
                progress_callback(15)

            end_np = resize_and_center_crop(end_image, width, height)
            Image.fromarray(end_np).save(os.path.join(output_path, f"{job_id}_end.png"))
            end_pt = (
                             torch.from_numpy(end_np).float() / 127.5 - 1
                     ).permute(2, 0, 1)[None, :, None]

        # -------------------------
        # VAE encoding
        # -------------------------
        job_status.message = "VAE encoding..."
        job_status.progress = 20
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(20)

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
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(25)

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
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(30)

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

        # Calculate sampling progress contribution

        # Total progress steps from all latent sections

        for pad_idx, pad in enumerate(latent_paddings):
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
                # Get denoised latents
                step = d.get('step', 0)

                # Process every 5th step
                if step % 5 == 0 or step == steps - 1:
                    try:
                        denoised = d['denoised']

                        # Create preview using vae_decode_fake (faster)
                        preview = vae_decode_fake(denoised)
                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                        preview_filename = os.path.join("uploads", f"{job_id}_latent_{step}.jpg")
                        # Save the latent as an image
                        Image.fromarray(preview).save(preview_filename)

                        # Update job status with latest latent preview
                        status_obj = job_statuses.get(job_id)
                        if status_obj:
                            logger.info(f"Updating job status with latent preview: {preview_filename}")
                            # Use a consistent path that doesn't include the step number
                            status_obj.current_latents = preview_filename
                            # Add a cache-busting parameter to force browser refresh
                            status_obj.current_latents += f"?step={step}&t={int(time.time())}"

                            # Check if this segment is already in the list (avoid duplicates)
                            segment_paths = status_obj.segments
                            segment_path = status_obj.current_latents

                            # Clear previous segments if this is the first step
                            if step == 0:
                                segment_paths = []

                            # Replace the last segment or add a new one
                            if segment_paths and segment_path.split('?')[0] == segment_paths[-1].split('?')[0]:
                                segment_paths[-1] = segment_path
                            else:
                                segment_paths.append(segment_path)

                            status_obj.segments = segment_paths

                            # Save updated status to disk
                            save_job_data(job_id, status_obj.to_dict())
                        else:
                            logger.info(f"Job status object not found for job_id {job_id}")
                    except Exception as e:
                        logger.info(f"Error saving latent preview: {e}")

                    # Check if the job has been cancelled
                    status_obj = job_statuses.get(job_id)
                    if status_obj and status_obj.status == "cancelled":
                        logger.info(f"Job {job_id} was cancelled during processing")
                        # This will cause the sampling to stop at the current step
                        d['stop'] = True

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
            save_bcthw_as_mp4(history_pixels, output_file, fps=30, crf=mp4_crf)
            # Unload VAE if not high VRAM
            if not high_vram:
                unload_complete_models(vae)

            job_status.segments.append(output_file)
            # Save updated status to disk
            save_job_data(job_id, job_status.to_dict())

            if is_last:
                break

        job_status.status = "completed"
        job_status.progress = 100
        job_status.message = "Generation completed"
        job_status.result_video = output_file
        logger.info(f"job_status: {job_status}")
        # Delete the latent preview image
        output_filename = f"uploads/{job_id}_latent.jpg"

        if os.path.exists(output_filename):
            os.remove(output_filename)
        # Save final status to disk
        save_job_data(job_id, job_status.to_dict())
        if progress_callback:
            progress_callback(100)
        logger.info(f"Generation completed")

    except Exception as e:
        job_status.status = "failed"
        job_status.message = str(e)
        # Save error status to disk
        save_job_data(job_id, job_status.to_dict())
        traceback.print_exc()
        if not os.path.exists(output_file):
            output_file = None

    if not high_vram and segment_index is None:
        unload_complete_models(
            text_encoder,
            text_encoder_2,
            image_encoder,
            vae,
            transformer
        )
    if not os.path.exists(output_file):
        output_file = None
    logger.info(f"Returning output_file: {output_file}")
    return output_file





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

    for segment in segments:
        segment_duration = get_segment_value(segment, 'duration', 1.0)
        segment_frames = int(segment_duration * 30)
        segment_steps = steps
        segment_workload = segment_steps * segment_frames
        segment_workloads.append(segment_workload)
        total_steps += segment_workload

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
                master_job_id=master_job_id,
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
            master_job_id=master_job_id,
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
            segment['image_path'] = segment['image_path'].replace("/uploads/", upload_path)
            # This shouldn't happen, but just in case, handler "\uploads\" to
            segment['image_path'] = segment['image_path'].replace("\\uploads\\", upload_path)
        request_dict['segments'] = segments
    return worker_multi_segment(**request_dict)
