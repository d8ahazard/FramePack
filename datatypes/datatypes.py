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

    def get_running_job_id(self):
        # Import from job_queue, and do it here to avoid circular import
        from handlers.job_queue import job_statuses
        # Get the first job in the queue that is running
        for job_id, status in job_statuses.items():
            if status.status == "running":
                return job_id
        return None

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        # If job_id is "undefined", and there's a running job in the queue, use that job_id
        if job_id == "undefined":
            job_id = self.get_running_job_id()
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


class ModuleJobSettings(BaseModel, ABC):
    """Base class for module-specific job settings"""
    pass


class FileExistsResponse(BaseModel):
    exists: bool
    path: Optional[str] = None


class PathsExistResponse(BaseModel):
    results: List[FileExistsResponse]


class SaveJobRequest(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result_video: str
    segments: List[str]
    current_latents: Optional[str] = None
    is_valid: bool
    missing_images: List[str]
    job_settings: Optional[dict] = None  # Dictionary containing all job settings including segment configurations
    queue_position: int
    created_timestamp: int

    def ensure_segments_in_job_settings(self):
        """
        Ensure that segments data is properly included in job_settings.
        This helps prevent issues when loading jobs to the timeline.
        """
        if not self.job_settings:
            return False

            # Check the format of job_settings
            if isinstance(self.job_settings, dict):
                # Handle the new dictionary structure with module name as key
                if any(isinstance(v, dict) for v in self.job_settings.values()):
                    # New format - find the module settings (e.g., 'framepack')
                    module_name = next(iter(self.job_settings.keys()))
                    module_settings = self.job_settings.get(module_name, {})

                    # Check if segments are in the module settings
                    if "segments" not in module_settings or not isinstance(module_settings["segments"], list):
                        return False

                    # Ensure we have the right number of segments
                    if len(self.segments) != len(module_settings["segments"]):
                        return False

                    return True
                else:
                    # Legacy format - direct settings object
                    # Check if segments are in job_settings
                    if "segments" not in self.job_settings or not isinstance(self.job_settings["segments"], list):
                        return False

                    # Ensure we have the right number of segments
                    if len(self.segments) != len(self.job_settings["segments"]):
                        return False

                    return True

            return False
        return None


class DeleteJobRequest(BaseModel):
    delete_images: bool = False


class DeleteVideoRequest(BaseModel):
    video_path: str


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage().find("206 Partial Content") >= 0:
            return False
        if record.getMessage().find("200 OK") >= 0:
            return False
        return True


class ErrorResponse(BaseModel):
    error: str


class GenerateResponse(BaseModel):
    job_id: str


class JobStatus:
    def __init__(self, job_id: str):
        self.job_id = job_id
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
        """Store the original job settings as a dictionary with module name as key"""
        self.job_settings = settings

        # For backward compatibility, check if settings has a direct segments field
        if settings and isinstance(settings, dict):
            # Check if it's already in the new format (dictionary with module names as keys)
            if any(isinstance(v, dict) for v in settings.values()):
                # It's already in the new format
                for module_name, module_settings in settings.items():
                    if isinstance(module_settings, dict) and "segments" in module_settings and isinstance(
                            module_settings["segments"], list):
                        # If we have segments in the stored settings, make sure they match
                        if len(self.segments) > 0 and len(module_settings["segments"]) != len(self.segments):
                            print(
                                f"Warning: Job has {len(self.segments)} segments but {module_name} settings has {len(module_settings['segments'])} segments")
            else:
                # Old format - segments at top level
                if "segments" in settings and isinstance(settings["segments"], list):
                    if len(self.segments) > 0 and len(settings["segments"]) != len(self.segments):
                        print(
                            f"Warning: Job has {len(self.segments)} segments but settings has {len(settings['segments'])} segments")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result_video: Optional[str] = None
    segments: List[str] = []
    current_latents: Optional[str] = None
    is_valid: bool = True  # Whether all images in job still exist
    missing_images: List[str] = []
    job_settings: Optional[dict] = None  # For storing original settings
    queue_position: int = 0  # Position in job queue
    created_timestamp: int = 0  # When the job was created


class SegmentConfig(BaseModel):
    image_path: str
    prompt: str
    duration: float
    
    def to_dict(self):
        """Convert the model to a dictionary for JSON serialization"""
        return {
            "image_path": self.image_path,
            "prompt": self.prompt,
            "duration": self.duration
        }
    
    def model_dump(self):
        """Alias for to_dict() for compatibility with Pydantic v2"""
        return self.to_dict()


class UploadResponse(BaseModel):
    success: bool
    filename: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None  # API URL to access the file
    upload_url: Optional[str] = None  # Static URL using /uploads/ path
    error: Optional[str] = None


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
