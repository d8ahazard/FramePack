from abc import ABC
import json
import logging
import time
from typing import List, Optional, Dict, Set

import torch
from pydantic import BaseModel
from starlette.websockets import WebSocket


class ConnectionManager:
    def __init__(self):
        # Store active connections by job_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, data: dict):
        if job_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_text(json.dumps(data))
                except Exception as e:
                    print(f"Error sending to websocket: {e}")
                    disconnected.add(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, job_id)

class ModuleJobSettings(ABC):
    pass

ModuleJobSettings.register(BaseModel)



class SaveJobRequest(BaseModel):
    job_id: str
    job_name: str
    status: str
    progress: int
    message: str
    result_video: str
    segments: List[str]
    current_latents: Optional[str] = None
    is_valid: bool
    missing_images: List[str]
    job_settings: Optional[dict] = None
    queue_position: int
    created_timestamp: int



class DeleteJobRequest(BaseModel):
    delete_images: bool = False


class DeleteVideoRequest(BaseModel):
    video_path: str


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.getMessage().find("GET /api/job_status/") >= 0)


class ErrorResponse(BaseModel):
    error: str


class GenerateResponse(BaseModel):
    job_id: str


class JobStatus:
    def __init__(self, job_id: str, job_name: str = None):
        self.job_id = job_id
        self.job_name = job_name
        self.status = "queued"  # status options: queued, running, completed, failed, cancelled, saved
        self.progress = 0
        self.message = ""
        self.result_video = None
        self.segments = []
        self.current_latents = None
        self.is_valid = True
        self.missing_images = []
        self.job_settings = None
        self.queue_position = 0  # Position in the job queue (0 = currently running)
        self.created_timestamp = int(time.time())  # Used for sorting

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result_video": self.result_video,
            "segments": self.segments,
            "current_latents": self.current_latents,
            "is_valid": self.is_valid,
            "missing_images": self.missing_images,
            "job_settings": self.job_settings,
            "queue_position": self.queue_position,
            "created_timestamp": self.created_timestamp
        }

    def set_job_settings(self, settings):
        """Store the original job settings"""
        self.job_settings = settings


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result_video: Optional[str] = None
    segments: List[str] = []
    current_latents: Optional[str] = None
    job_name: Optional[str] = None
    is_valid: bool = True  # Whether all images in job still exist
    missing_images: List[str] = []
    job_settings: Optional[dict] = None  # For storing original settings
    queue_position: int = 0  # Position in job queue
    created_timestamp: int = 0  # When the job was created


class SegmentConfig(BaseModel):
    image_path: str
    prompt: str
    duration: float


class UploadResponse(BaseModel):
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None


class VideoRequest(BaseModel):
    global_prompt: str
    negative_prompt: str
    segments: List[SegmentConfig]
    job_name: Optional[str] = None  # Optional job name for user reference
    seed: int = 31337
    steps: int = 25
    guidance_scale: float = 10.0
    use_teacache: bool = True
    enable_adaptive_memory: bool = True
    resolution: int = 640
    mp4_crf: int = 16
    gpu_memory_preservation: float = 6.0
    include_last_frame: bool = False  # Control whether to generate a segment for the last frame


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return
