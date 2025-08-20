"""
Feature Extraction Module using ArcFace
Converts face images to embedding vectors
"""

import cv2
import numpy as np
from typing import List, Union
import logging
import os

# Try to import insightface, fallback to a mock implementation if not available
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Using mock implementation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Feature extractor using ArcFace model"""
    
    def __init__(self, model_name: str = 'buffalo_l', ctx_id: int = -1):
        """
        Initialize ArcFace model
        
        Args:
            model_name: Name of the InsightFace model
            ctx_id: Context ID (-1 for CPU, 0+ for GPU)
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        
        if INSIGHTFACE_AVAILABLE:
            self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        else:
            self.app = None
            logger.warning("Using mock feature extractor")
    
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 512-dimensional embedding from face image
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            512-dimensional embedding vector
        """
        if not INSIGHTFACE_AVAILABLE or self.app is None:
            # Mock implementation for testing
            return np.random.randn(512).astype(np.float32)
        
        try:
            # Get face analysis
            faces = self.app.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected in image for feature extraction")
                return np.zeros(512, dtype=np.float32)
            
            # Use the first detected face
            face = faces[0]
            embedding = face.normed_embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def extract_features_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from multiple face images
        
        Args:
            face_images: List of face images
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, face_image in enumerate(face_images):
            embedding = self.extract_features(face_image)
            embeddings.append(embedding)
            logger.debug(f"Extracted features for face {i+1}/{len(face_images)}")
            
        return embeddings
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """
        Extract features from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            512-dimensional embedding vector
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        return self.extract_features(image)


class MockFeatureExtractor:
    """Mock feature extractor for testing when InsightFace is not available"""
    
    def __init__(self, *args, **kwargs):
        logger.info("Using mock feature extractor")
        
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """Return random 512-dimensional vector"""
        return np.random.randn(512).astype(np.float32)
        
    def extract_features_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Return list of random vectors"""
        return [self.extract_features(img) for img in face_images]
        
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Return random vector"""
        return np.random.randn(512).astype(np.float32)
