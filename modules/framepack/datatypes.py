from typing import Dict, List, Any

from datatypes.datatypes import ModuleJobSettings, SegmentConfig


class FramePackJobSettings(ModuleJobSettings):
    job_id: str
    global_prompt: str
    mp4_crf: int
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
    include_last_frame: bool = False  # Control whether to generate a segment for the last frame


