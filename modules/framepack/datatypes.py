import os
from typing import Dict, List, Any, Optional, Union

from datatypes.datatypes import ModuleJobSettings, SegmentConfig


class FramePackJobSettings(ModuleJobSettings):
    job_id: str
    global_prompt: Union[str, List[str], None] = None
    mp4_crf: int
    negative_prompt: str
    segments: List[SegmentConfig]
    seed: int = 31337
    steps: int = 25
    guidance_scale: float = 10.0
    use_teacache: bool = True
    enable_adaptive_memory: bool = True
    resolution: int = 720
    mp4_crf: int = 16
    gpu_memory_preservation: float = 6.0
    include_last_frame: bool = False  # Control whether to generate a segment for the last frame
    auto_prompt: bool = False
    fps: int = 30  # Added FPS control
    lora_model: Optional[str] = None  # Path to LORA model
    lora_scale: float = 1.0  # Scale for LORA influence
    final_image: Optional[str] = None  # Optional final image path

