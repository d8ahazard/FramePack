from typing import Dict, List, Any, Optional, Union

from datatypes.datatypes import ModuleJobSettings, SegmentConfig


class WanJobSettings(ModuleJobSettings):
    job_id: str
    prompt: str
    task: str = "i2v-14B"  # Auto-detected based on inputs if not specified
    size: str = "1280*720"
    frame_num: int = 81
    fps: int = 16  # Frames per second in output video
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    base_seed: int = -1  # Random seed for generation
    
    # Segments for consistent handling with FramePack
    segments: Optional[List[Union[SegmentConfig, Dict[str, Any]]]] = None
    
    # Image-to-Video fields
    image: Optional[str] = None  # Input image path for I2V
    
    # First-Last-Frame-to-Video fields
    first_frame: Optional[str] = None  # First frame path for FLF2V
    last_frame: Optional[str] = None  # Last frame path for FLF2V
    
    # Sampling parameters
    sample_steps: int = 50  # Diffusion sampling steps
    sample_guide_scale: float = 5.0  # Guidance scale
    sample_shift: float = 5.0  # Shift for flow matching
    
    # Prompt expansion settings
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: Optional[str] = None
    prompt_extend_target_lang: str = "zh"
    
    # Advanced settings
    t5_cpu: bool = False  # Whether to place T5 model on CPU
    offload_model: bool = True  # Whether to offload model to CPU during generation 