from typing import Dict, List, Any

from datatypes.datatypes import ModuleJobSettings


class FramePackJobSettings(ModuleJobSettings):
    enable_adaptive_memory: bool
    global_prompt: str
    gpu_memory_preservation: float
    guidance_scale: float
    include_last_frame: bool
    job_name: str
    mp4_crf: int
    negative_prompt: str
    resolution: int
    seed: int
    steps: int
    segments: List[Dict[str, Any]]
    use_teacache: bool
