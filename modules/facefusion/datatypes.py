from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from datatypes.datatypes import ModuleJobSettings

# Type aliases for clarity
VisionFrame = np.ndarray
BoundingBox = np.ndarray  # [x1, y1, x2, y2]
FaceLandmark5 = np.ndarray  # 5 key facial points
FaceLandmark68 = np.ndarray  # 68 facial landmarks
Matrix = np.ndarray
Mask = np.ndarray
Score = float

class FaceSwapperModel(str, Enum):
    """Available face swapper models"""
    INSWAPPER = "inswapper"
    SIMSWAP = "simswap"
    BLENDSWAP = "blendswap"

class FaceEnhancerModel(str, Enum):
    """Available face enhancer models"""
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"
    RESTOREFORMER = "restoreformer"

class FaceDetectorModel(str, Enum):
    """Available face detector models"""
    RETINAFACE = "retinaface"
    SCRFD = "scrfd"
    YOLOFACE = "yoloface"

class FaceLandmarkerModel(str, Enum):
    """Available face landmark models"""
    DFAN2 = "2dfan4"
    FAN68 = "fan_68"

@dataclass
class Face:
    """Represents a detected face with its features"""
    bounding_box: BoundingBox
    landmark_5: FaceLandmark5
    landmark_68: Optional[FaceLandmark68] = None
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    age: Optional[range] = None
    gender: Optional[str] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the face"""
        x1, y1, x2, y2 = self.bounding_box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def size(self) -> float:
        """Get the size (max dimension) of the face"""
        x1, y1, x2, y2 = self.bounding_box
        return max(x2 - x1, y2 - y1)

@dataclass
class ProcessOptions:
    """Options for face processing"""
    face_swapper_model: FaceSwapperModel = FaceSwapperModel.INSWAPPER
    face_enhancer_model: Optional[FaceEnhancerModel] = FaceEnhancerModel.GFPGAN
    face_detector_model: FaceDetectorModel = FaceDetectorModel.RETINAFACE
    face_landmarker_model: FaceLandmarkerModel = FaceLandmarkerModel.DFAN2
    face_enhancer_blend: int = 80  # 0-100 blend factor for enhancement
    face_swap_threshold: float = 0.7  # Threshold for face similarity
    enhance_size_threshold: int = 400  # Minimum face size for enhancement
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

@dataclass
class ProcessResult:
    """Result of face swapping/enhancing process"""
    success: bool
    output_path: Optional[Path] = None
    frames_processed: int = 0
    faces_swapped: int = 0
    faces_enhanced: int = 0
    error: Optional[str] = None

class FaceFusionJobSettings(ModuleJobSettings):
    """Job settings for FaceFusion module"""
    source_image_path: str
    target_video_path: str
    output_path: str
    face_swapper_model: str = FaceSwapperModel.INSWAPPER.value
    face_enhancer_model: Optional[str] = FaceEnhancerModel.GFPGAN.value
    face_detector_model: str = FaceDetectorModel.RETINAFACE.value
    face_enhancer_blend: int = 80
    enhance_size_threshold: int = 400
    face_swap_threshold: float = 0.7
    
    def to_process_options(self) -> ProcessOptions:
        """Convert job settings to ProcessOptions"""
        return ProcessOptions(
            face_swapper_model=FaceSwapperModel(self.face_swapper_model),
            face_enhancer_model=FaceEnhancerModel(self.face_enhancer_model) if self.face_enhancer_model else None,
            face_detector_model=FaceDetectorModel(self.face_detector_model),
            face_enhancer_blend=self.face_enhancer_blend,
            enhance_size_threshold=self.enhance_size_threshold,
            face_swap_threshold=self.face_swap_threshold
        )
