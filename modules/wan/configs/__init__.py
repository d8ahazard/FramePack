"""Configuration for Wan models."""
import os
from dataclasses import dataclass
import torch
from typing import List, Tuple, Dict, Any

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@dataclass
class WanConfig:
    """Base configuration for Wan models."""
    name: str
    # Model parameters
    dim: int = 1536
    z_dim: int = 16
    num_layers: int = 30
    num_heads: int = 12
    patch_size: Tuple[int, int, int] = (1, 16, 16)
    window_size: List[int] = None
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    # Process parameters
    text_len: int = 64
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "umt5-xxl"  # Will be resolved to google/umt5-xxl or checkpoint_dir/google/umt5-xxl
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    clip_checkpoint: str = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    clip_tokenizer: str = "xlm-roberta-large"  # Will be resolved to the HF model or local path
    num_train_timesteps: int = 1000
    t5_dtype: torch.dtype = torch.bfloat16
    clip_dtype: torch.dtype = torch.bfloat16
    param_dtype: torch.dtype = torch.bfloat16
    sample_fps: int = 16
    sample_neg_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# FLF2V configuration based on i2v-14B
flf2v_14B = WanConfig(
    name="Config: Wan FLF2V 14B",
    dim=5120,
    num_layers=40,
    num_heads=40,
)
flf2v_14B.sample_neg_prompt = "镜头切换，" + flf2v_14B.sample_neg_prompt

# I2V configuration
i2v_14B = WanConfig(
    name="Config: Wan I2V 14B",
    dim=5120,
    num_layers=40,
    num_heads=40,
)

# Export configurations
WAN_CONFIGS = {
    'flf2v-14B': flf2v_14B,
    'i2v-14B': i2v_14B
}

# Size configurations
SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
} 