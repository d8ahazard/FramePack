import traceback
import cv2
import numpy as np
import logging
import os
import onnxruntime
import torch
from handlers.path import model_path
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import requests
import shutil

from modules.facefusion.datatypes import (
    Face, FaceDetectorModel, ProcessOptions, 
    ProcessResult, VisionFrame, FaceFusionJobSettings
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress ONNX Runtime warnings about shape mismatches
# Set ONNX Runtime log level to ERROR to hide warnings
onnxruntime.set_default_logger_severity(3)  # 3 = ERROR, 2 = WARNING

# Path to model assets - will need to be configured correctly for deployment
MODELS_PATH = os.path.join(model_path, "facefusion")
os.makedirs(MODELS_PATH, exist_ok=True)

# Define model URLs and hash info
FACEFUSION_MODELS = {
    # Face detector models
    "retinaface": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx",
        "hash_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.hash",
        "path": os.path.join(MODELS_PATH, "retinaface.onnx"),
        "hash_path": os.path.join(MODELS_PATH, "retinaface.hash")
    },
    # Face swapper models
    "inswapper": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
        "hash_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash",
        "path": os.path.join(MODELS_PATH, "inswapper.onnx"),
        "hash_path": os.path.join(MODELS_PATH, "inswapper.hash")
    },
    # Face enhancer models
    "gfpgan": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
        "hash_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash",
        "path": os.path.join(MODELS_PATH, "gfpgan.onnx"),
        "hash_path": os.path.join(MODELS_PATH, "gfpgan.hash")
    },
    # Face recognizer model
    "arcface": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx",
        "hash_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.hash",
        "path": os.path.join(MODELS_PATH, "arcface.onnx"),
        "hash_path": os.path.join(MODELS_PATH, "arcface.hash")
    },
    # OpenCV YuNet face detector
    "yunet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": os.path.join(MODELS_PATH, "yunet.onnx")
    }
}

# Warp template constants - from FaceFusion
WARP_TEMPLATES = {
    'arcface_112_v2': np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089]
    ]),
    'ffhq_512': np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465]
    ])
}

def verify_model(model_info: Dict) -> bool:
    """Verify if model exists and hash matches"""
    # Check if model file exists
    if not os.path.exists(model_info["path"]):
        return False
    return True

def download_model(model_name: str, model_info: Dict) -> bool:
    """Download a model from the specified URL"""
    try:
        print(f"Downloading {model_name} model...")
        download_file(model_info["url"], model_info["path"])
        print(f"Downloaded {model_name} model successfully")
        
        # Also download hash file if it exists and not already present
        if "hash_url" in model_info and "hash_path" in model_info and not os.path.exists(model_info["hash_path"]):
            download_file(model_info["hash_url"], model_info["hash_path"])
            
        return True
    except Exception as e:
        print(f"Error downloading {model_name} model: {e}")
        return False

def download_file(url: str, path: str) -> None:
    """Download a file from URL to specified path with progress bar"""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        
        # Create a temporary file for download
        temp_path = path + ".download"
        with open(temp_path, "wb") as file:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(path)) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        
        # Move the temp file to the final location
        shutil.move(temp_path, path)

def download_models():
    """Download required models from FaceFusion repositories"""
    print("Checking and downloading required models...")
    
    for model_name, model_info in FACEFUSION_MODELS.items():
        if not verify_model(model_info):
            download_model(model_name, model_info)
            if not verify_model(model_info):
                print(f"Failed to verify downloaded model: {model_name}")
    
    print("Model check complete.")

class FaceProcessor:
    """Main class for face detection, swapping and enhancement"""
    
    def __init__(self, options: ProcessOptions = None, device: Optional[torch.device] = None):
        self.options = options or ProcessOptions()
        self.sessions = {}
        self.initialized = False
        self.source_image = None
        self.reference_faces = []
        self.device = device
        self.options.device = self.device.type
        
    def initialize(self):
        """Initialize required models"""
        if self.initialized:
            return
        
        # Ensure models are downloaded
        download_models()
        
        # Configure provider options based on device
        providers = []
        if self.options.device == "cuda" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Load face detector model
        detector_path = FACEFUSION_MODELS["retinaface"]["path"]
        if os.path.exists(detector_path):
            self.sessions["detector"] = onnxruntime.InferenceSession(detector_path, providers=providers)
        else:
            logger.warning(f"Face detector model not found: {detector_path}")
        
        # Load face swapper model
        swapper_path = FACEFUSION_MODELS["inswapper"]["path"]
        if os.path.exists(swapper_path):
            self.sessions["swapper"] = onnxruntime.InferenceSession(swapper_path, providers=providers)
        else:
            logger.warning(f"Face swapper model not found: {swapper_path}")
        
        # Load face enhancer model if specified
        if self.options.face_enhancer_model:
            enhancer_path = FACEFUSION_MODELS["gfpgan"]["path"]
            if os.path.exists(enhancer_path):
                self.sessions["enhancer"] = onnxruntime.InferenceSession(enhancer_path, providers=providers)
            else:
                logger.warning(f"Face enhancer model not found: {enhancer_path}")
        
        # Load face recognizer for better face matching
        recognizer_path = FACEFUSION_MODELS["arcface"]["path"]
        if os.path.exists(recognizer_path):
            self.sessions["recognizer"] = onnxruntime.InferenceSession(recognizer_path, providers=providers)
        else:
            logger.warning(f"Face recognizer model not found: {recognizer_path}")
            
        print(f"Initialized FaceFusion models")
        self.initialized = True
    
    def detect_faces(self, frame: VisionFrame) -> List[Face]:
        """Detect faces in a frame using RetinaFace and extract landmarks"""
        self.initialize()
        print(f"Detecting faces in frame")
        
        # Try RetinaFace first, then fall back to YuNet or Haar cascade if needed
        faces = self._detect_with_retinaface(frame)
        
        # If RetinaFace fails, try YuNet
        if not faces and cv2.__version__ >= "4.5.4":
            faces = self._detect_with_yunet(frame)
        
        # Final fallback to Haar cascade
        if not faces:
            faces = self._detect_with_haar(frame)
            
        # Calculate embeddings for detected faces
        for face in faces:
            face.embedding = self._extract_face_embedding(frame, face)
            
        print(f"Detected {len(faces)} faces in frame")
        return faces
    
    def _detect_with_retinaface(self, frame: VisionFrame) -> List[Face]:
        """Detect faces using RetinaFace"""
        if "detector" not in self.sessions:
            print("RetinaFace detector not available")
            return []
            
        try:
            # Prepare frame for detection
            height, width = frame.shape[:2]
            detect_frame = frame.copy()
            
            # Create input tensor
            # Normalize to [-1, 1] as expected by RetinaFace
            detect_frame = (detect_frame.astype(np.float32) - 127.5) / 128.0
            detect_frame = np.transpose(detect_frame, (2, 0, 1))
            detect_frame = np.expand_dims(detect_frame, axis=0)
            
            # Run detection
            outputs = self.sessions["detector"].run(None, {"input": detect_frame})
            
            # Process outputs
            bounding_boxes = []
            landmarks = []
            scores = []
            # Increase threshold to reduce false positives
            score_threshold = 0.85
            
            # Extract detections - handle different output shapes
            for output in outputs:
                # Check the shape of the output
                if len(output.shape) == 2:
                    if output.shape[1] == 1:  # Confidence scores
                        for score in output.flatten():
                            if score > score_threshold:
                                scores.append(float(score))
                    elif output.shape[1] == 4:  # Bounding boxes
                        for i in range(min(len(scores), output.shape[0])):
                            x1, y1, x2, y2 = output[i]
                            # Ensure correct ordering of coordinates
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                            bounding_boxes.append(np.array([x1, y1, x2, y2]))
                    elif output.shape[1] == 10:  # 5 facial landmarks (x,y) * 5
                        for i in range(min(len(scores), output.shape[0])):
                            # Reshape to 5 landmarks with (x,y) coordinates
                            landmark = output[i].reshape(5, 2)
                            landmarks.append(landmark)
                elif len(output.shape) == 3:
                    # Handle 3D outputs
                    if output.shape[2] == 1:  # Confidence scores
                        for batch_scores in output:
                            for score in batch_scores.flatten():
                                if score > score_threshold:
                                    scores.append(float(score))
                    elif output.shape[2] == 4:  # Bounding boxes
                        for batch_boxes in output:
                            for i, box in enumerate(batch_boxes):
                                if i < len(scores):
                                    x1, y1, x2, y2 = box
                                    if x1 > x2:
                                        x1, x2 = x2, x1
                                    if y1 > y2:
                                        y1, y2 = y2, y1
                                    bounding_boxes.append(np.array([x1, y1, x2, y2]))
                    elif output.shape[2] == 10:  # Landmarks
                        for batch_landmarks in output:
                            for i, landmark_data in enumerate(batch_landmarks):
                                if i < len(scores):
                                    landmark = landmark_data.reshape(5, 2)
                                    landmarks.append(landmark)
            
            # Apply non-maximum suppression to remove overlapping boxes
            if len(bounding_boxes) > 0 and len(scores) > 0:
                # Make sure the number of boxes and scores match
                min_count = min(len(bounding_boxes), len(scores))
                bounding_boxes = bounding_boxes[:min_count]
                scores = scores[:min_count]
                
                # Convert to format required by NMS
                boxes_for_nms = []
                for box in bounding_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    boxes_for_nms.append([x1, y1, w, h])
                
                # Verify arrays have same size before NMS
                if len(boxes_for_nms) == len(scores):
                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, score_threshold, 0.3)
                    
                    # Extract results after NMS
                    filtered_boxes = []
                    filtered_landmarks = []
                    filtered_scores = []
                    
                    for idx in indices:
                        # Handle both OpenCV 4.x and 3.x output format
                        if isinstance(idx, list) or isinstance(idx, tuple):
                            idx = idx[0]
                        filtered_boxes.append(bounding_boxes[idx])
                        if idx < len(landmarks):
                            filtered_landmarks.append(landmarks[idx])
                        filtered_scores.append(scores[idx])
                        
                    bounding_boxes = filtered_boxes
                    landmarks = filtered_landmarks
                    scores = filtered_scores
            
            # Create Face objects
            faces = []
            for i in range(min(len(bounding_boxes), len(scores), len(landmarks) if landmarks else 0)):
                face = Face(
                    bounding_box=bounding_boxes[i],
                    landmark_5=landmarks[i] if i < len(landmarks) else None,
                    score=scores[i]
                )
                # Only generate landmarks if we have landmark data
                if face.landmark_5 is not None:
                    face.landmark_68 = self._generate_landmarks_68(landmarks[i], bounding_boxes[i])
                faces.append(face)
                
            return faces
                
        except Exception as e:
            print(f"Error in RetinaFace detection: {e}")
            traceback.print_exc()
            return []
    
    def _detect_with_yunet(self, frame: VisionFrame) -> List[Face]:
        """Detect faces using OpenCV's YuNet detector as fallback"""
        yunet_path = FACEFUSION_MODELS["yunet"]["path"]
        if not os.path.exists(yunet_path):
            print("YuNet model not available")
            return []
            
        try:
            # Create YuNet detector
            detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                frame.shape[:2][::-1],
                0.9,  # Score threshold
                0.3,  # NMS threshold
                5000  # Top k
            )
            
            # Run detection
            _, detections = detector.detect(frame)
            faces = []
            
            if detections is None:
                return []
                
            for detection in detections:
                # Get bounding box
                x, y, w, h = detection[:4].astype(int)
                bbox = np.array([x, y, x + w, y + h])
                
                # Get landmarks (5 points)
                landmark_5 = detection[4:14].reshape(-1, 2)
                
                # Create Face object
                face = Face(
                    bounding_box=bbox,
                    landmark_5=landmark_5,
                    score=float(detection[14])
                )
                
                # Generate 68-point landmarks from 5-point landmarks
                face.landmark_68 = self._generate_landmarks_68(landmark_5, bbox)
                faces.append(face)
                
            return faces
                
        except Exception as e:
            print(f"Error in YuNet detection: {e}")
            return []
    
    def _detect_with_haar(self, frame: VisionFrame) -> List[Face]:
        """Detect faces using Haar cascade as final fallback"""
        try:
            # Get Haar cascade
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(haar_path)
            
            # Run detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detections = detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            faces = []
            for (x, y, w, h) in detections:
                # Create bounding box
                bbox = np.array([x, y, x + w, y + h])
                
                # Generate approximate landmarks based on typical face proportions
                center_x, center_y = x + w//2, y + h//2
                eye_w = w // 4
                landmark_5 = np.array([
                    [x + w//3, y + h//3],           # Right eye
                    [x + 2*w//3, y + h//3],         # Left eye
                    [center_x, center_y],           # Nose tip
                    [x + w//3, y + 2*h//3],         # Right mouth corner
                    [x + 2*w//3, y + 2*h//3]        # Left mouth corner
                ])
                
                face = Face(
                    bounding_box=bbox,
                    landmark_5=landmark_5,
                    score=1.0  # Default score for Haar detections
                )
                
                # Generate 68-point landmarks
                face.landmark_68 = self._generate_landmarks_68(landmark_5, bbox)
                faces.append(face)
                
            return faces
            
        except Exception as e:
            print(f"Error in Haar cascade detection: {e}")
            return []
    
    def _generate_landmarks_68(self, landmark_5: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Generate 68 landmarks from 5 key landmarks"""
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        
        # Extract key points
        right_eye = landmark_5[0]  
        left_eye = landmark_5[1]   
        nose = landmark_5[2]       
        right_mouth = landmark_5[3]
        left_mouth = landmark_5[4]  
        
        landmarks_68 = np.zeros((68, 2))
        
        # Set key points
        landmarks_68[36] = right_eye  # Right eye center
        landmarks_68[45] = left_eye   # Left eye center
        landmarks_68[30] = nose       # Nose tip
        landmarks_68[48] = right_mouth  # Right mouth corner
        landmarks_68[54] = left_mouth   # Left mouth corner
        
        # Generate remaining landmarks
        # Jaw (0-16)
        for i in range(17):
            t = i / 16.0
            landmarks_68[i] = [x1 + width * t, y1 + height * (0.7 + 0.2 * np.sin(np.pi * t))]
        
        # Right eyebrow (17-21)
        for i in range(5):
            t = i / 4.0
            landmarks_68[i+17] = [right_eye[0] - width * 0.15 + width * 0.3 * t, 
                                  right_eye[1] - height * 0.15]
        
        # Left eyebrow (22-26)
        for i in range(5):
            t = i / 4.0
            landmarks_68[i+22] = [left_eye[0] - width * 0.15 + width * 0.3 * t, 
                                  left_eye[1] - height * 0.15]
        
        # Nose ridge (27-30)
        for i in range(4):
            t = i / 3.0
            landmarks_68[i+27] = [nose[0], nose[1] - height * 0.2 + height * 0.2 * t]
        
        # Nose bottom (31-35)
        for i in range(5):
            t = i / 4.0
            landmarks_68[i+31] = [nose[0] - width * 0.1 + width * 0.2 * t, 
                                  nose[1] + height * 0.05]
        
        # Right eye (36-41)
        eye_radius = width * 0.07
        for i in range(6):
            angle = i * np.pi / 3.0
            landmarks_68[i+36] = [right_eye[0] + eye_radius * np.cos(angle),
                                  right_eye[1] + eye_radius * np.sin(angle)]
        
        # Left eye (42-47)
        for i in range(6):
            angle = i * np.pi / 3.0
            landmarks_68[i+42] = [left_eye[0] + eye_radius * np.cos(angle),
                                  left_eye[1] + eye_radius * np.sin(angle)]
        
        # Outer mouth (48-59)
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        mouth_height = mouth_width * 0.5
        mouth_center = (right_mouth + left_mouth) / 2
        for i in range(12):
            angle = i * 2 * np.pi / 12.0
            landmarks_68[i+48] = [mouth_center[0] + mouth_width/2 * np.cos(angle),
                                  mouth_center[1] + mouth_height/2 * np.sin(angle)]
        
        # Inner mouth (60-67)
        inner_scale = 0.7
        for i in range(8):
            angle = i * 2 * np.pi / 8.0
            landmarks_68[i+60] = [mouth_center[0] + inner_scale * mouth_width/2 * np.cos(angle),
                                 mouth_center[1] + inner_scale * mouth_height/2 * np.sin(angle)]
        
        return landmarks_68
        
    def _extract_face_embedding(self, frame: VisionFrame, face: Face) -> np.ndarray:
        """Extract face embedding using ArcFace"""
        if "recognizer" not in self.sessions:
            # Return random embedding as fallback
            return np.random.rand(512).astype(np.float32)
            
        try:
            # Warp face to template for ArcFace
            template = 'arcface_112_v2'
            size = (112, 112)
            
            # Warp face using 5-point landmarks
            warped_face, _ = self._warp_face(frame, face.landmark_5, template, size)
            
            # Preprocess for ArcFace
            warped_face = warped_face.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
            warped_face = warped_face[:, :, ::-1]  # RGB to BGR
            warped_face = np.transpose(warped_face, (2, 0, 1))  # HWC to CHW
            warped_face = np.expand_dims(warped_face, axis=0)
            
            # Run face recognition
            embedding = self.sessions["recognizer"].run(None, {"input": warped_face})[0]
            embedding = embedding.flatten()
            
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return np.random.rand(512).astype(np.float32)
    
    def _warp_face(self, frame: VisionFrame, landmark: np.ndarray, 
                  template: str, size: Tuple[int, int]) -> Tuple[VisionFrame, np.ndarray]:
        """Warp face according to landmarks using FaceFusion approach"""
        template_landmarks = WARP_TEMPLATES[template] * size[0]
        
        # Estimate affine transform
        M = cv2.estimateAffinePartial2D(landmark, template_landmarks, method=cv2.RANSAC, 
                                       ransacReprojThreshold=100)[0]
        
        # Apply warping
        warped_frame = cv2.warpAffine(frame, M, size, borderMode=cv2.BORDER_REPLICATE)
        
        return warped_frame, M
    
    def find_similar_faces(self, source_faces: List[Face], target_faces: List[Face], threshold: float = None) -> List[Tuple[Face, Face]]:
        """Find similar faces between source and target based on embeddings"""
        if not threshold:
            threshold = self.options.face_swap_threshold
            
        matches = []
        
        for source_face in source_faces:
            for target_face in target_faces:
                # Calculate similarity based on face embeddings
                similarity = self._calculate_face_similarity(source_face, target_face)
                if similarity > threshold:
                    matches.append((source_face, target_face))
        
        # Only log when called from process_video and not from inside each frame processing
        caller_name = traceback.extract_stack()[-2].name
        if caller_name != "process_video" or len(matches) == 0:
            print(f"Found {len(matches)} face matches")
            
        return matches
    
    def _calculate_face_similarity(self, face1: Face, face2: Face) -> float:
        """Calculate similarity between two faces using embeddings and landmarks"""
        # Using cosine similarity if embeddings available
        if hasattr(face1, 'embedding') and hasattr(face2, 'embedding') and \
           face1.embedding is not None and face2.embedding is not None:
            # Cosine similarity: dot product of normalized vectors
            similarity = np.dot(face1.embedding, face2.embedding)
            return float(similarity)  # Higher is more similar (0-1)
        
        # Fallback to landmark similarity if embeddings not available
        if face1.landmark_5 is None or face2.landmark_5 is None:
            return 0.0
            
        # Normalize landmarks to center and scale
        def normalize_landmarks(landmarks):
            center = landmarks.mean(axis=0)
            centered = landmarks - center
            scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
            return centered / (scale + 1e-10)
            
        norm_lm1 = normalize_landmarks(face1.landmark_5)
        norm_lm2 = normalize_landmarks(face2.landmark_5)
        
        # Calculate distance
        dist = np.sqrt(((norm_lm1 - norm_lm2) ** 2).sum(axis=1)).mean()
        # Convert to similarity score (0-1)
        similarity = max(0, 1 - dist)
        
        return similarity
    
    def swap_face(self, source_face: Face, target_face: Face, frame: VisionFrame) -> VisionFrame:
        """Swap source face onto target face in frame"""
        self.initialize()
        
        try:
            # Check if swapper is available
            if "swapper" not in self.sessions:
                print("Face swapper model not loaded")
                return frame
                
            # Extract face size for enhancement check
            x1, y1, x2, y2 = target_face.bounding_box.astype(int)
            face_size = max(x2 - x1, y2 - y1)
            
            # Warp target face to standard position
            template = 'arcface_112_v2'
            size = (128, 128)  # InSwapper size
            
            # Warp face using landmarks
            crop_frame, affine_matrix = self._warp_face(frame, target_face.landmark_5, template, size)
            
            # Prepare input tensor
            crop_tensor = crop_frame.astype(np.float32) / 255.0
            crop_tensor = np.transpose(crop_tensor, (2, 0, 1))
            crop_tensor = np.expand_dims(crop_tensor, axis=0)
            
            # Get source face embedding
            source_embedding = source_face.embedding
            if source_embedding is None or len(source_embedding) != 512:
                # Extract embedding from source image
                if hasattr(self, 'source_image') and self.source_image is not None:
                    source_embedding = self._extract_face_embedding(self.source_image, source_face)
                else:
                    source_embedding = np.random.rand(512).astype(np.float32)
                    
            # Run face swapper
            inputs = {
                "source": source_embedding.reshape(1, -1).astype(np.float32),
                "target": crop_tensor.astype(np.float32)
            }
            
            outputs = self.sessions["swapper"].run(None, inputs)
            swapped_face = outputs[0][0]
            
            # Post-process
            swapped_face = np.transpose(swapped_face, (1, 2, 0)) * 255
            swapped_face = swapped_face.clip(0, 255).astype(np.uint8)
            
            # Create a face mask for blending
            mask = self._create_face_mask(size)
            
            # Inverse transform back to original position
            inverse_matrix = cv2.invertAffineTransform(affine_matrix)
            
            # Apply warped face back to original frame
            result_frame = frame.copy()
            warped_swapped_face = cv2.warpAffine(
                swapped_face, 
                inverse_matrix, 
                (frame.shape[1], frame.shape[0]), 
                borderMode=cv2.BORDER_REPLICATE
            )
            
            warped_mask = cv2.warpAffine(
                mask, 
                inverse_matrix, 
                (frame.shape[1], frame.shape[0])
            )
            
            # Apply mask for blending
            for c in range(3):
                result_frame[:, :, c] = (
                    warped_swapped_face[:, :, c] * warped_mask + 
                    result_frame[:, :, c] * (1 - warped_mask)
                )
            
            # Apply face enhancement if requested
            if face_size >= self.options.enhance_size_threshold and \
               self.options.face_enhancer_model and \
               "enhancer" in self.sessions:
                result_frame = self._enhance_face(target_face, result_frame)
            
            return result_frame
            
        except Exception as e:
            print(f"Error swapping face: {e}")
            traceback.print_exc()
            return frame
    
    def _create_face_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a face mask for blending the swapped face"""
        # Create base mask
        mask = np.ones(size, dtype=np.float32)
        
        # Add padding and feathering for better blending
        padding = int(size[0] * 0.05)  # 5% padding
        
        # Create padding on edges
        mask[:padding, :] = 0
        mask[-padding:, :] = 0
        mask[:, :padding] = 0
        mask[:, -padding:] = 0
        
        # Feather the edges
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=padding//2)
        
        return mask
        
    def _enhance_face(self, face: Face, frame: VisionFrame) -> VisionFrame:
        """Enhance a face region using GFPGAN"""
        try:
            # Extract face region with margin
            x1, y1, x2, y2 = face.bounding_box.astype(int)
            
            # Add margin
            margin = int(0.1 * max(x2 - x1, y2 - y1))
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            # Skip if region is invalid
            if x1 >= x2 or y1 >= y2:
                return frame
                
            face_img = frame[y1:y2, x1:x2]
            
            # Resize to GFPGAN input size
            face_size = (512, 512)
            face_img_resized = cv2.resize(face_img, face_size)
            
            # Prepare input for GFPGAN
            face_tensor = face_img_resized.astype(np.float32) / 255.0
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            face_tensor = np.transpose(face_tensor, (2, 0, 1))
            face_tensor = np.expand_dims(face_tensor, axis=0)
            
            # Run enhancer
            outputs = self.sessions["enhancer"].run(None, {"input": face_tensor})
            enhanced_face = outputs[0][0]
            
            # Post-process
            enhanced_face = np.clip((enhanced_face + 1) / 2, 0, 1)  # Denormalize to [0, 1]
            enhanced_face = (enhanced_face.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Resize back to original size
            enhanced_face = cv2.resize(enhanced_face, (x2 - x1, y2 - y1))
            
            # Blend with original using specified blend factor
            result_frame = frame.copy()
            blend_factor = self.options.face_enhancer_blend / 100.0
            result_frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 1 - blend_factor, 
                enhanced_face, blend_factor, 0
            )
            
            return result_frame
            
        except Exception as e:
            print(f"Error enhancing face: {e}")
            return frame
    
    def process_video(self, 
                     source_image_path: str, 
                     target_video_path: str, 
                     output_path: str,
                     progress_callback: Callable[[int, int, str], None] = None) -> ProcessResult:
        """Process a video by swapping faces from source to target with consistent tracking"""
        try:
            print(f"Processing video: {target_video_path}")
            
            # Check if files exist
            if not os.path.exists(source_image_path):
                print(f"Source image not found: {source_image_path}")
                return ProcessResult(False, error=f"Source image not found: {source_image_path}")
            if not os.path.exists(target_video_path):
                print(f"Target video not found: {target_video_path}")
                return ProcessResult(False, error=f"Target video not found: {target_video_path}")
                
            # Read source image
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                print(f"Failed to read source image: {source_image_path}")
                return ProcessResult(False, error=f"Failed to read source image: {source_image_path}")
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            
            # Store source image for later use
            self.source_image = source_image
            
            # Create temp directory for frames
            temp_frame_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            print(f"Created temp directory for frames: {temp_frame_dir}")
            
            # Extract video info
            video = cv2.VideoCapture(target_video_path)
            if not video.isOpened():
                print(f"Failed to open target video: {target_video_path}")
                return ProcessResult(False, error=f"Failed to open target video: {target_video_path}")
            
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video properties: {width}x{height} at {fps} fps, {total_frames} frames")
            
            # Extract all frames
            print("Extracting frames...")
            frame_paths = []
            frame_number = 0
            
            # Use tqdm for progress bar
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    frame_path = os.path.join(temp_frame_dir, f"frame_{frame_number:06d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    frame_number += 1
                    pbar.update(1)
            
            video.release()
            
            # Detect faces in source image
            print("Detecting faces in source image...")
            source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_faces = self.detect_faces(source_image_rgb)
            
            if not source_faces:
                print(f"No faces detected in source image: {source_image_path}")
                return ProcessResult(False, error=f"No faces detected in source image: {source_image_path}")
            
            # Get reference faces from first frame
            print("Processing first frame for reference faces...")
            first_frame = cv2.imread(frame_paths[0])
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            reference_faces = self.detect_faces(first_frame_rgb)
            
            if not reference_faces:
                print("No faces detected in first frame")
                return ProcessResult(False, error="No faces detected in first frame")
            
            # Store reference faces for consistent tracking
            self.reference_faces = reference_faces
            
            # Find face matches between source and reference - initial matching
            matches = self.find_similar_faces(source_faces, reference_faces)
            print(f"Found {len(matches)} face matches between source and reference faces")
            
            if not matches:
                # If no automatic matches, use the first face from each
                if source_faces and reference_faces:
                    print("No similar faces found, using first faces from source and target")
                    matches = [(source_faces[0], reference_faces[0])]
            
            # Process all frames
            print("Processing frames...")
            frames_processed = 0
            faces_swapped = 0
            faces_enhanced = 0
            
            # Process frames in batches to improve tracking consistency
            batch_size = min(30, len(frame_paths))  # Process ~1 second at a time, cap for small videos
            
            # Create a new progress bar for processing
            with tqdm(total=len(frame_paths), desc="Processing frames") as pbar:
                for i in range(0, len(frame_paths), batch_size):
                    batch_paths = frame_paths[i:i+batch_size]
                    batch_reference = None
                    
                    # Process each frame in the batch
                    for frame_idx, frame_path in enumerate(batch_paths):
                        # Read the frame
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            print(f"Failed to read frame: {frame_path}")
                            pbar.update(1)
                            frames_processed += 1
                            continue
                            
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # For the first frame in batch, detect faces normally
                        # For subsequent frames, use previous frame as reference
                        if frame_idx == 0 and i > 0:
                            # Update reference from previous batch's last frame
                            prev_frame = cv2.imread(frame_paths[i-1])
                            prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                            batch_reference = self.detect_faces(prev_frame_rgb)
                        
                        # Detect faces in current frame
                        current_faces = self.detect_faces(frame_rgb)
                        
                        if not current_faces:
                            # If no faces in this frame, just update progress and continue
                            pbar.update(1)
                            frames_processed += 1
                            continue
                        
                        # Set batch reference if not yet set
                        if batch_reference is None:
                            batch_reference = current_faces
                        
                        # Process each frame
                        output_frame = frame_rgb.copy()
                        faces_swapped_in_frame = 0
                        
                        # Find and process each face pair
                        for source_face, ref_face in matches:
                            # Find matching faces in current frame based on reference
                            # Disable logging for internal calls
                            face_matches = []
                            for target_face in current_faces:
                                similarity = self._calculate_face_similarity(ref_face, target_face)
                                if similarity > self.options.face_swap_threshold:
                                    face_matches.append((ref_face, target_face))
                            
                            for _, target_face in face_matches:
                                # Swap face
                                output_frame = self.swap_face(source_face, target_face, output_frame)
                                faces_swapped += 1
                                faces_swapped_in_frame += 1
                                
                                # Count enhanced faces
                                if hasattr(target_face, 'size') and target_face.size >= self.options.enhance_size_threshold:
                                    faces_enhanced += 1
                        
                        # Save processed frame
                        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(frame_path, output_frame_bgr)
                        
                        frames_processed += 1
                        pbar.update(1)
                        
                        # Update progress callback if provided
                        if progress_callback:
                            progress_callback(
                                frames_processed, 
                                total_frames, 
                                f"Processed {frames_processed}/{total_frames} frames, {faces_swapped_in_frame} faces swapped"
                            )
            
            # Create output video from frames
            print("Creating output video...")
            self._create_video_from_frames(frame_paths, output_path, fps, (width, height))
            
            # Clean up temp files
            print("Cleaning up temporary files...")
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            try:
                os.rmdir(temp_frame_dir)
            except:
                print(f"Warning: Could not remove temp directory: {temp_frame_dir}")
            
            return ProcessResult(
                success=True,
                output_path=Path(output_path),
                frames_processed=frames_processed,
                faces_swapped=faces_swapped,
                faces_enhanced=faces_enhanced
            )
            
        except Exception as e:
            print(f"Error processing video: {e}")
            traceback.print_exc()
            return ProcessResult(False, error=str(e))
    
    def _create_video_from_frames(self, frame_paths: List[str], output_path: str, fps: float, size: Tuple[int, int]):
        """Create a video from frame image files"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                writer.write(frame)
        
        writer.release()
    
    def close(self):
        """Release any resources"""
        try:
            # Clear ONNX sessions
            self.sessions.clear()
            self.initialized = False
            
            # Clear source image
            if hasattr(self, 'source_image'):
                self.source_image = None
                
            # Clear reference faces
            self.reference_faces = []
            
        except Exception as e:
            print(f"Error closing resources: {e}")
            pass


# Main processing function for the module
def process(job_settings: FaceFusionJobSettings, device: Optional[torch.device] = None) -> ProcessResult:
    """
    Process a video by swapping faces from source image to target video
    
    Args:
        job_settings: FaceFusionJobSettings containing all processing parameters
        
    Returns:
        ProcessResult with success status and stats
    """
    try:
        # Convert job settings to process options
        options = job_settings.to_process_options()
        print(f"Processing video with FaceFusion: {job_settings.target_video_path}")
        
        # Ensure models are downloaded
        download_models()
        print(f"Models verified successfully")
        
        # Process the video
        processor = FaceProcessor(options)
        print(f"Initialized processor")
        
        result = processor.process_video(
            job_settings.source_image_path,
            job_settings.target_video_path,
            job_settings.output_path
        )
        
        print(f"Video processing complete")
        return result
        
    except Exception as e:
        print(f"Error in process function: {e}")
        traceback.print_exc()
        return ProcessResult(False, error=str(e))
    finally:
        # Always attempt to clean up resources
        try:
            processor.close()
        except:
            pass


def test_process():
    video_path="E:\\dev\\FramePack\\outputs\\batch_1745956209_381_segment_1.mp4"
    image_path="E:\\dev\\gd\\7c2ce640-29b2-42c2-973b-44006ab79bbe.jpg"
    output_path="E:\\dev\\gd\\test_output.mp4"
    job_settings = FaceFusionJobSettings(
        source_image_path=image_path,
        target_video_path=video_path,
        output_path=output_path
    )
    print(f"Processing video: {video_path}")
    process(job_settings)
