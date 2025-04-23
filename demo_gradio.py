import os
import re
import PIL
import subprocess
import shutil
import traceback
import argparse
import time

import gradio as gr
import torch
import einops
import numpy as np
from huggingface_hub import snapshot_download

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
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


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
# Argument parsing / HF cache location
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--preload", action='store_true', help="Preload all models at startup")
parser.add_argument("--hf_token", type=str, help="Hugging Face authentication token for private models")
args = parser.parse_args()

# Don't override HF_HOME if already set
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(
        os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download'))
    )

print(args)

# Optional: preload all models at startup
if args.preload:
    model_paths = preload_all_models(use_auth_token=args.hf_token)
    hunyuan_path = model_paths["hunyuan"]
    flux_path = model_paths["flux"]
    framepack_path = model_paths["framepack"]
else:
    # Download and get paths for model repositories
    hunyuan_path = check_download_model("hunyuanvideo-community/HunyuanVideo", use_auth_token=args.hf_token)
    flux_path = check_download_model("lllyasviel/flux_redux_bfl", use_auth_token=args.hf_token)
    framepack_path = check_download_model("lllyasviel/FramePackI2V_HY", use_auth_token=args.hf_token)

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

stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# ----------------------------------------
# Worker: single-segment generation
# ----------------------------------------
@torch.no_grad()
def worker(
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
    global adaptive_memory_management
    adaptive_memory_management = enable_adaptive_memory

    # how many latent sections
    total_latent_sections = int(
        max(round((total_second_length * 30) / (latent_window_size * 4)), 1)
    )

    # decide job_id + base_name
    if segment_index is not None and master_job_id:
        job_id = master_job_id
        segment_number = segment_index + 1
    else:
        job_id = generate_timestamp()
        segment_number = 1

    base_name = f"{job_id}_segment_{segment_number}"
    output_path = os.path.join(outputs_folder, f"{base_name}.mp4")
    temp_dir = os.path.join(outputs_folder, f"{base_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    stream.output_queue.push(
        ('progress', (None, '', make_progress_bar_html(0, 'Starting ...')))
    )

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
        stream.output_queue.push(
            ('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...')))
        )

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
        stream.output_queue.push(
            ('progress', (None, '', make_progress_bar_html(0, 'Image processing ...')))
        )

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_np = resize_and_center_crop(input_image, width, height)
        Image.fromarray(input_np).save(os.path.join(outputs_folder, f"{job_id}.png"))
        input_pt = (
            torch.from_numpy(input_np).float() / 127.5 - 1
        ).permute(2, 0, 1)[None, :, None]

        has_end_image = end_image is not None
        if has_end_image:
            stream.output_queue.push(
                ('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...')))
            )
            end_np = resize_and_center_crop(end_image, width, height)
            Image.fromarray(end_np).save(os.path.join(outputs_folder, f"{job_id}_end.png"))
            end_pt = (
                torch.from_numpy(end_np).float() / 127.5 - 1
            ).permute(2, 0, 1)[None, :, None]

        # -------------------------
        # VAE encoding
        # -------------------------
        stream.output_queue.push(
            ('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...')))
        )

        if vae not in models_to_keep_in_memory:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_pt, vae)
        if has_end_image:
            end_latent = vae_encode(end_pt, vae)

        # -------------------------
        # CLIP Vision encoding
        # -------------------------
        stream.output_queue.push(
            ('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...')))
        )

        
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
        stream.output_queue.push(
            ('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...')))
        )

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

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

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

            # ALWAYS KEEP TRANSFORMER ON GPU
            # if not high_vram:
            #     unload_complete_models()
            #     move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # re-init teaCache each loop
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = vae_decode_fake(d['denoised'])
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                step = d['i'] + 1
                pct = int(100.0 * step / steps)
                desc = (
                    f"Generated {max(0, (total_generated * 4 - 3))} frames, "
                    f"{(max(0, (total_generated * 4 - 3)) / 30):.2f}s so far"
                )
                stream.output_queue.push(
                    ('progress', (preview, desc, make_progress_bar_html(pct, f"Sampling {step}/{steps}")))
                )
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

            # if not high_vram:
            #     offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)

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
            stream.output_queue.push(('file', output_path))

            if is_last:
                break

    except Exception:
        traceback.print_exc()
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

    stream.output_queue.push(('end', None))
    return


# ----------------------------------------
# Worker: keyframe-driven multi-segment
# ----------------------------------------
@torch.no_grad()
def worker_keyframe(
    input_image,
    end_image,
    keyframes,
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
    resolution
):
    global stream, adaptive_memory_management
    adaptive_memory_management = enable_adaptive_memory

    # sort keyframes
    keyframe_data = []
    for kf in keyframes or []:
        fn = os.path.basename(kf)
        m = re.search(r'_(\d+)\.', fn)
        if m:
            keyframe_data.append((int(m.group(1)), kf))
        else:
            print(f"Skipping invalid keyframe name: {fn}")
    keyframe_data.sort(key=lambda x: x[0])
    sorted_kfs = [p for _, p in keyframe_data]

    if not sorted_kfs:
        return worker(
            input_image,
            end_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg, gs, rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            enable_adaptive_memory,
            resolution
        )

    # load images
    imgs = []
    for p in sorted_kfs:
        try:
            with PIL.Image.open(p) as img:
                imgs.append(np.array(img.convert('RGB')))
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if not imgs:
        return worker(
            input_image,
            end_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg, gs, rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            enable_adaptive_memory,
            resolution
        )

    all_frames = [input_image] + imgs
    if end_image is not None:
        all_frames.append(end_image)

    num_segments = len(all_frames) - 1
    seconds_per_segment = round(total_second_length / num_segments, 2)

    master_job_id = generate_timestamp()
    master_temp = os.path.join(outputs_folder, f"{master_job_id}_temp")
    os.makedirs(master_temp, exist_ok=True)

    segment_paths = []

    # process backwards
    for i in range(num_segments - 1, -1, -1):
        seg_no = num_segments - i
        start_f = all_frames[i]
        end_f = all_frames[i + 1]

        stream.output_queue.push((
            'progress', (
                None, '',
                make_progress_bar_html(
                    0,
                    f"Segment {seg_no}/{num_segments} (backwards)..."
                )
            )
        ))

        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            return

        # maybe clear cache
        curr_free = get_cuda_free_memory_gb(gpu)
        if not high_vram and curr_free < 2.0:
            torch.cuda.empty_cache()
            if curr_free < 1.0:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

        # call worker for this segment
        worker(
            start_f,
            end_f,
            prompt,
            n_prompt,
            seed,
            seconds_per_segment,
            latent_window_size,
            steps,
            cfg, gs, rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            enable_adaptive_memory,
            resolution,
            segment_index=i,
            master_job_id=master_job_id
        )

        time.sleep(0.5)
        seg_path = os.path.join(
            outputs_folder,
            f"{master_job_id}_segment_{seg_no}.mp4"
        )
        segment_paths.append((seg_no, seg_path))

    # concat in correct order
    concat_txt = os.path.join(master_temp, "concat_list.txt")
    with open(concat_txt, 'w') as f:
        for seg_no, path in sorted(segment_paths, key=lambda x: x[0]):
            f.write(f"file '{path}'\n")

    final_out = os.path.join(outputs_folder, f"{master_job_id}.mp4")
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', concat_txt, '-c', 'copy', final_out
    ], check=True)

    if not high_vram:
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae, transformer
        )
    try:
        shutil.rmtree(master_temp)
    except Exception as e:
        print(f"Cleanup error: {e}")

    stream.output_queue.push(('file', final_out))
    stream.output_queue.push(('end', None))
    return


# ----------------------------------------
# Process wrapper & UI hookup
# ----------------------------------------
def process(
    input_image,
    end_image,
    keyframes,
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
    resolution
):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    load_models()
    stream = AsyncStream()

    if keyframes and len(keyframes) > 0:
        async_run(
            worker_keyframe,
            input_image,
            end_image,
            keyframes,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg, gs, rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            enable_adaptive_memory,
            resolution
        )
    else:
        async_run(
            worker,
            input_image,
            end_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg, gs, rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            enable_adaptive_memory,
            resolution
        )

    current_video = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            current_video = data
            yield current_video, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'end':
            yield current_video, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


# ----------------------------------------
# Gradio layout
# ----------------------------------------
quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

css = make_progress_bar_css()
block = gr.Blocks(css=css)

with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    sources='upload',
                    type="numpy",
                    label="Start Image",
                    height=320
                )
                end_image = gr.Image(
                    sources='upload',
                    type="numpy",
                    label="End Image (Optional)",
                    height=320
                )
            keyframes = gr.File(
                label="Keyframes (Optional)",
                file_count="multiple",
                file_types=["image"],
                height=50
            )
            gr.Markdown(
                "*Keyframe images should be named with _n suffix where n is the frame order (0, 1, 2...). Files will be sorted by this number.*",
                elem_id="keyframe-info"
            )
            resolution = gr.Slider(
                label="Resolution",
                minimum=240,
                maximum=720,
                value=640,
                step=16
            )
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(
                samples=quick_prompts,
                label='Quick List',
                samples_per_page=1000,
                components=[prompt]
            )
            example_quick_prompts.click(
                lambda x: x[0],
                inputs=[example_quick_prompts],
                outputs=prompt,
                show_progress=False,
                queue=False
            )

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_teacache = gr.Checkbox(
                        label='Use TeaCache',
                        value=True,
                        info='Faster speed, but slightly lower detail.'
                    )
                    enable_adaptive_memory = gr.Checkbox(
                        label='Enable Adaptive Memory Management',
                        value=True,
                        info='Better for long videos on limited VRAM.'
                    )

                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="",
                    visible=False
                )
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(
                    label="Total Video Length (Seconds)",
                    minimum=1,
                    maximum=600,
                    value=5,
                    step=0.1
                )
                latent_window_size = gr.Slider(
                    label="Latent Window Size",
                    minimum=1,
                    maximum=33,
                    value=9,
                    step=1,
                    visible=False
                )
                steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=100,
                    value=25,
                    step=1
                )
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=32.0,
                    value=1.0,
                    step=0.01,
                    visible=False
                )
                gs = gr.Slider(
                    label="Distilled CFG Scale",
                    minimum=1.0,
                    maximum=32.0,
                    value=10.0,
                    step=0.01
                )
                rs = gr.Slider(
                    label="CFG Re-Scale",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    visible=False
                )
                gpu_memory_preservation = gr.Slider(
                    label="GPU Inference Preserved Memory (GB)",
                    minimum=6,
                    maximum=128,
                    value=6,
                    step=0.1
                )
                mp4_crf = gr.Slider(
                    label="MP4 Compression",
                    minimum=0,
                    maximum=100,
                    value=16,
                    step=1,
                    info="Lower = higher quality; use 16 if you see black frames."
                )

        with gr.Column():            
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            preview_image = gr.Image(
                label="Next Latents",
                height=200,
                visible=False
            )
            result_video = gr.Video(
                label="Finished Frames",
                autoplay=True,
                show_share_button=False,
                height=512,
                loop=True
            )
            gr.Markdown(
                'Note: ending actions are generated before start ones due to reverse sampling.'
            )
            

    gr.HTML(
        '<div style="text-align:center;margin-top:20px;">'
        'Share results on '
        '<a href="https://x.com/search?q=framepack&f=live" target="_blank">'
        'FramePack Twitter Thread</a></div>'
    )

    inputs = [
        input_image,
        end_image,
        keyframes,
        prompt,
        n_prompt,
        seed,
        total_second_length,
        latent_window_size,
        steps,
        cfg, gs, rs,
        gpu_memory_preservation,
        use_teacache,
        mp4_crf,
        enable_adaptive_memory,
        resolution
    ]
    start_button.click(
        fn=process,
        inputs=inputs,
        outputs=[
            result_video,
            preview_image,
            progress_desc,
            progress_bar,
            start_button,
            end_button
        ]
    )
    end_button.click(fn=end_process)

block.queue()
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
