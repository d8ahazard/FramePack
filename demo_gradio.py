from diffusers_helper.hf_login import login

import os
import re
import PIL

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import shutil

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 32  # Raised from 60GB to 24GB as a more realistic threshold for high-end consumer cards
adaptive_memory_management = True  # Default to True for better experience on most systems

print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')
print(f'Adaptive Memory Management: {adaptive_memory_management}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# Always enable slicing and tiling for VAE regardless of VRAM to improve reliability
vae.enable_slicing()
vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

def log_vram(model):
    current_free_mem_gb = get_cuda_free_memory_gb(gpu)
    try:
        model.to(gpu)
        used_mem_gb = current_free_mem_gb - get_cuda_free_memory_gb(gpu)
        print(f'{model.__class__.__name__} used {used_mem_gb:.2f} GB VRAM')
        model.to('cpu')
    except Exception as e:
        print(f'Error logging VRAM for {model.__class__.__name__}: {e}')
        model.to('cpu')

if not high_vram:
    # models_to_test = [transformer, text_encoder, text_encoder_2, image_encoder, vae]
    # for model in models_to_test:
    #     log_vram(model)
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution, out_file=None):
    # Set the adaptive memory flag based on user selection
    global adaptive_memory_management
    adaptive_memory_management = enable_adaptive_memory
    
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    # Create a single master output filename
    master_output_filename = out_file if out_file else os.path.join(outputs_folder, f'{job_id}_master.mp4')
    # Create a temp directory for update files
    temp_dir = os.path.join(outputs_folder, f'{job_id}_temp')
    os.makedirs(temp_dir, exist_ok=True)

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Initial model loading strategy based on video length and available memory
        models_to_keep_in_memory = []
        # HunyuanVideoTransformer3DModelPacked used 22.75 GB VRAM - transformer
        # LlamaModel used 2.50 GB VRAM - text_encoder
        # CLIPTextModel used 0.23 GB VRAM - text_encoder_2
        # SiglipVisionModel used 0.83 GB VRAM - image_encoder
        # AutoencoderKLHunyuanVideo used 0.45 GB VRAM
        
        # Intelligent memory management based on available VRAM
        if high_vram or adaptive_memory_management:
            # Define memory tiers based on actual model sizes with safety margins
            if free_mem_gb >= 26:  # Ultra high memory - keep everything (RTX 3090/4090)
                text_encoder.to(gpu)
                text_encoder_2.to(gpu)
                image_encoder.to(gpu)
                vae.to(gpu)
                transformer.to(gpu)
                models_to_keep_in_memory = [text_encoder, text_encoder_2, image_encoder, vae, transformer]
                print("Ultra high memory mode: Keeping all models in memory")
                
            elif free_mem_gb >= 23:  # High memory - keep transformer and small models
                text_encoder.to(gpu)
                text_encoder_2.to(gpu)
                transformer.to(gpu)
                models_to_keep_in_memory = [text_encoder, text_encoder_2, transformer]
                print("High memory mode: Keeping transformer and text encoders in memory")
                
            elif free_mem_gb >= 4:  # Medium memory - keep only smaller models
                text_encoder.to(gpu)
                text_encoder_2.to(gpu)
                models_to_keep_in_memory = [text_encoder, text_encoder_2]
                
                # Optionally keep image_encoder and VAE if we have enough space
                if free_mem_gb >= 6:
                    image_encoder.to(gpu)
                    vae.to(gpu)
                    models_to_keep_in_memory.extend([image_encoder, vae])
                    print("Medium memory mode: Keeping text encoders, image encoder and VAE in memory")
                else:
                    print("Medium memory mode: Keeping only text encoders in memory")
            else:
                print("Low memory mode: No models preloaded")
        else:
            # Clean GPU for maximum compatibility mode
            print("Compatibility mode: Adaptive memory management disabled")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if text_encoder not in models_to_keep_in_memory:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Clean up text encoders if they're not in our keep list and we need memory
        if text_encoder not in models_to_keep_in_memory and text_encoder_2 not in models_to_keep_in_memory:
            unload_complete_models(text_encoder, text_encoder_2)

        # Processing input image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # Process end image if provided
        has_end_image = end_image is not None
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if vae not in models_to_keep_in_memory:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if image_encoder not in models_to_keep_in_memory:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings or use a weighted approach
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # Ensure transformer is loaded if not already in memory
        if transformer not in models_to_keep_in_memory:
            # Calculate appropriate preserved memory based on other models and TeaCache usage
            preserved_memory = gpu_memory_preservation
            if use_teacache:
                # TeaCache needs more memory, so preserve more
                preserved_memory = max(gpu_memory_preservation, 8)
            
            move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)
            
        # Initialize TeaCache once outside the loop if possible
        if use_teacache:
            # If memory is tight, we need to be more careful with TeaCache
            current_free_mem = get_cuda_free_memory_gb(gpu)
            if current_free_mem < 5.0:
                print(f"Low memory for TeaCache ({current_free_mem:.2f} GB free), using reduced memory settings")
                transformer.initialize_teacache(enable_teacache=True, num_steps=min(steps, 20))
            else:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
			
			            
            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)


            # Only load transformer if not in keep list and not currently loaded
            if transformer not in models_to_keep_in_memory and not transformer.device == gpu:
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
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

            # Smart model management for VAE decoding
            need_vae = True
            
            # Check if we need to offload transformer to make room for VAE
            if need_vae and vae not in models_to_keep_in_memory:
                # Only offload transformer if not in keep list and there's not enough memory for both
                if transformer not in models_to_keep_in_memory:
                    # Check available memory before deciding to offload
                    current_free_mem = get_cuda_free_memory_gb(gpu)
                    if current_free_mem < 1.0:  # If memory is tight, offload transformer
                        print(f"Offloading transformer to CPU due to memory pressure. Current free memory: {current_free_mem:.2f} GB")
                        offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=1)
                    load_model_as_complete(vae, target_device=gpu)
                else:
                    # If transformer stays in memory, just load VAE alongside it
                    load_model_as_complete(vae, target_device=gpu, unload=False)
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # Only unload VAE if not in keep list and we need memory
            if vae not in models_to_keep_in_memory:
                current_free_mem = get_cuda_free_memory_gb(gpu)
                # Only unload if memory is tight (under 2GB) and not the last section
                if current_free_mem < 2.0 and not is_last_section:
                    unload_complete_models(vae)

            # Always save to the master output file
            save_bcthw_as_mp4(history_pixels, master_output_filename, fps=30, crf=mp4_crf)
            
            # Create a unique copy for this update to force UI refresh
            update_filename = os.path.join(temp_dir, f'update_{len(latent_paddings) - latent_padding}.mp4')
            save_bcthw_as_mp4(history_pixels, update_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            # Always push the update file to the UI
            stream.output_queue.push(('file', update_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

    finally:
        # Clean up at the end
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        
        # Create a final copy for the completed video
        final_output_filename = os.path.join(outputs_folder, f'{job_id}_final.mp4')
        if os.path.exists(master_output_filename):
            shutil.copy2(master_output_filename, final_output_filename)
            stream.output_queue.push(('file', final_output_filename))
            
            # Delete initialization images if requested by caller
            img_path = os.path.join(outputs_folder, f'{job_id}.png')
            end_img_path = os.path.join(outputs_folder, f'{job_id}_end.png')
            if os.path.exists(img_path) and out_file:
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Error removing initialization image: {e}")
            if os.path.exists(end_img_path) and out_file:
                try:
                    os.remove(end_img_path)
                except Exception as e:
                    print(f"Error removing end image: {e}")
            
        # Clean up temporary files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

    stream.output_queue.push(('end', None))
    return final_output_filename


@torch.no_grad()
def worker_keyframe(input_image, end_image, keyframes, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution):
    """
    Process video generation with multiple keyframes.
    Each keyframe becomes a target for a segment of the video.
    Video segments are generated in reverse order, with frames being appended to the master video.
    """
    global stream, adaptive_memory_management

    # Sort keyframes by their numerical suffix
    sorted_keyframes = []
    if keyframes and len(keyframes) > 0:
        # Parse keyframe order from filenames
        keyframe_data = []
        for kf_path in keyframes:
            # Extract number from filename (e.g., image_3.jpg -> 3)
            filename = os.path.basename(kf_path)
            match = re.search(r'_(\d+)\.[^.]+$', filename)
            if match:
                frame_num = int(match.group(1))
                keyframe_data.append((frame_num, kf_path))
            else:
                print(f"Warning: Keyframe {filename} doesn't follow the naming convention (should have _n suffix). Skipping.")
        
        # Sort by frame number
        keyframe_data.sort(key=lambda x: x[0])
        sorted_keyframes = [kf_path for _, kf_path in keyframe_data]
    
    # If no valid keyframes found, fall back to regular worker
    if not sorted_keyframes:
        print("No valid keyframes found, using regular worker")
        return worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, 
                     latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                     use_teacache, mp4_crf, enable_adaptive_memory, resolution)
    
    print(f"Processing with {len(sorted_keyframes)} keyframes")
    
    # Load all keyframe images
    keyframe_images = []
    for kf_path in sorted_keyframes:
        try:
            with PIL.Image.open(kf_path) as img:
                # Convert to RGB and numpy array
                img = img.convert('RGB')
                keyframe_images.append(np.array(img))
        except Exception as e:
            print(f"Error loading keyframe {kf_path}: {e}")
            continue
    
    # If we couldn't load any keyframes, fall back to regular worker
    if not keyframe_images:
        print("Failed to load keyframe images, using regular worker")
        return worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, 
                     latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                     use_teacache, mp4_crf, enable_adaptive_memory, resolution)
    
    # All frames in sequence: start frame, keyframes, end frame (if provided)
    all_frames = [input_image] + keyframe_images
    if end_image is not None:
        all_frames.append(end_image)
    
    print(f"Total frames for keyframe processing: {len(all_frames)}")
    
    # Calculate time per segment
    num_segments = len(all_frames) - 1
    # Need some kind of rounding as we are doing 30 FPS
    # 10 seconds is 300 frames, so we can round to 0.1 seconds  
    seconds_per_segment = round(total_second_length / num_segments, 2)
    
    # Create a job ID for the entire sequence
    master_job_id = generate_timestamp()
    master_output_filename = os.path.join(outputs_folder, f'{master_job_id}_master.mp4')
    master_temp_dir = os.path.join(outputs_folder, f'{master_job_id}_temp')
    os.makedirs(master_temp_dir, exist_ok=True)
    
    final_output_filename = None
    
    # Process segments in REVERSE order (from last to first)
    # Iterate from num_segments-1 down to 0
    for i in range(num_segments-1, -1, -1):
        # Get the frames for this segment
        start_frame = all_frames[i]
        end_frame = all_frames[i+1]
        segment_length = seconds_per_segment
        
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Processing segment {num_segments-i}/{num_segments} (working backwards)...'))))
        
        # Check if user wants to stop
        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            return
        
        # Only clear GPU memory if we're low on VRAM
        current_free_mem = get_cuda_free_memory_gb(gpu)
        if not high_vram and current_free_mem < 2.0:
            print(f"Low memory detected ({current_free_mem:.2f} GB free), clearing GPU cache")
            torch.cuda.empty_cache()
            # Only unload models if memory is critically low
            if current_free_mem < 1.0:
                print("Critical memory situation, unloading all models")
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        else:
            print(f"Sufficient memory available ({current_free_mem:.2f} GB free), keeping loaded models")
        
        # Process this segment, using the master output filename
        print(f"Starting segment {num_segments-i}/{num_segments}, working backwards: {i} to {i+1}, length: {segment_length:.2f} seconds")
        output_file = worker(start_frame, end_frame, prompt, n_prompt, seed, segment_length, 
              latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
              use_teacache, mp4_crf, enable_adaptive_memory, resolution,
              out_file=master_output_filename)
        
        # For UI updates, use the file from the last segment
        if i == 0:
            final_output_filename = output_file
            # Force UI update with the final video
            stream.output_queue.push(('file', final_output_filename))
        
        # Wait for the worker to finish and update the UI
        import time
        time.sleep(0.5)
    
    # Final cleanup if needed
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    
    # Create the final output video 
    if final_output_filename is None:
        final_output_filename = os.path.join(outputs_folder, f'{master_job_id}_final.mp4')
        if os.path.exists(master_output_filename):
            shutil.copy2(master_output_filename, final_output_filename)
            # Update the UI with the final video
            stream.output_queue.push(('file', final_output_filename))
            
    # Delete the master file since we have a final copy
    if os.path.exists(master_output_filename):
        try:
            os.remove(master_output_filename)
        except Exception as e:
            print(f"Error removing master file: {e}")
    
    # Clean up temporary directory
    try:
        if os.path.exists(master_temp_dir):
            shutil.rmtree(master_temp_dir)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
    # Send the final end signal to the stream to complete the process
    stream.output_queue.push(('end', None))
    return


def process(input_image, end_image, keyframes, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    # Determine whether to use keyframe worker or regular worker
    if keyframes and len(keyframes) > 0:
        # For keyframes, we need to run synchronously to avoid memory issues
        async_run(worker_keyframe, input_image, end_image, keyframes, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution)
    else:
        async_run(worker, input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution)

    current_video_file = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            current_video_file = data
            # Update the UI with the new video file
            yield current_video_file, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            # Return the last video file
            yield current_video_file, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(sources='upload', type="numpy", label="Start Image", height=320)
                end_image = gr.Image(sources='upload', type="numpy", label="End Image (Optional)", height=320)
            keyframes = gr.File(label="Keyframes (Optional)", file_count="multiple", file_types=["image"], height=50)
            gr.Markdown("*Keyframe images should be named with _n suffix where n is the frame order (0, 1, 2...). Files will be sorted by this number.*", elem_id="keyframe-info")
            resolution = gr.Slider(label="Resolution", minimum=240, maximum=720, value=640, step=16)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    enable_adaptive_memory = gr.Checkbox(label='Enable Adaptive Memory Management', value=True, info='Reduces loading/offloading of models for faster long video generation.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=600, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, end_image, keyframes, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, enable_adaptive_memory, resolution]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
