"""Face Recognition on Edge - A comprehensive face recognition pipeline"""

from .face_detector import FaceDetector
from .feature_extractor import FeatureExtractor
from .face_matcher import FaceMatcher
from .face_recognition_pipeline import FaceRecognitionPipeline
from .lfw_dataset import LFWDataset
from .deepface_pipeline import DeepFacePipeline

__version__ = "0.1.0"
__all__ = [
    "FaceDetector",
    "FeatureExtractor",
    "FaceMatcher",
    "FaceRecognitionPipeline",
    "LFWDataset",
    "DeepFacePipeline"
]
