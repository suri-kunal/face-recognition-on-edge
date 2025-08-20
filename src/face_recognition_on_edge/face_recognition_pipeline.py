"""
Complete Face Recognition Pipeline
Integrates all components: detection, feature extraction, and matching
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
from pathlib import Path

from .face_detector import FaceDetector
from .feature_extractor import FeatureExtractor
from .face_matcher import FaceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """Complete face recognition pipeline"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.5,
                 detection_confidence: float = 0.5,
                 target_face_size: Tuple[int, int] = (112, 112)):
        """
        Initialize the complete pipeline
        
        Args:
            similarity_threshold: Threshold for face matching
            detection_confidence: Confidence threshold for face detection
            target_face_size: Target size for cropped faces
        """
        self.face_detector = FaceDetector(min_detection_confidence=detection_confidence)
        self.feature_extractor = FeatureExtractor()
        self.face_matcher = FaceMatcher(similarity_threshold=similarity_threshold)
        self.target_face_size = target_face_size
        
        logger.info("Face recognition pipeline initialized")
    
    def process_group_photo(self, group_photo_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process group photo to extract faces and their embeddings
        
        Args:
            group_photo_path: Path to group photo
            
        Returns:
            Tuple of (cropped_faces, face_embeddings)
        """
        logger.info(f"Processing group photo: {group_photo_path}")
        
        # Detect and crop faces
        cropped_faces = self.face_detector.detect_and_crop_faces(
            group_photo_path, self.target_face_size
        )
        
        if not cropped_faces:
            logger.warning("No faces detected in group photo")
            return [], []
        
        # Extract features from all faces
        face_embeddings = self.feature_extractor.extract_features_batch(cropped_faces)
        
        logger.info(f"Extracted features from {len(face_embeddings)} faces")
        return cropped_faces, face_embeddings
    
    def process_probe_image(self, probe_image_path: str) -> np.ndarray:
        """
        Process probe image to extract embedding
        
        Args:
            probe_image_path: Path to probe image
            
        Returns:
            Probe embedding vector
        """
        logger.info(f"Processing probe image: {probe_image_path}")
        
        # Load and resize probe image
        probe_image = cv2.imread(probe_image_path)
        if probe_image is None:
            raise ValueError(f"Could not load probe image from {probe_image_path}")
        
        # Resize to target size
        probe_resized = cv2.resize(probe_image, self.target_face_size)
        
        # Extract features
        probe_embedding = self.feature_extractor.extract_features(probe_resized)
        
        logger.info("Extracted features from probe image")
        return probe_embedding
    
    def recognize_face(self, probe_image_path: str, group_photo_path: str) -> Dict:
        """
        Complete face recognition pipeline
        
        Args:
            probe_image_path: Path to probe image
            group_photo_path: Path to group photo
            
        Returns:
            Dictionary with recognition results
        """
        logger.info("Starting face recognition pipeline")
        
        try:
            # Process group photo
            cropped_faces, candidate_embeddings = self.process_group_photo(group_photo_path)
            
            if not candidate_embeddings:
                return {
                    'success': False,
                    'error': 'No faces detected in group photo',
                    'num_candidates': 0
                }
            
            # Process probe image
            probe_embedding = self.process_probe_image(probe_image_path)
            
            # Perform matching
            match_results = self.face_matcher.match_with_details(
                probe_embedding, candidate_embeddings
            )
            
            # Add additional information
            match_results.update({
                'success': True,
                'probe_image': probe_image_path,
                'group_photo': group_photo_path,
                'num_candidates': len(candidate_embeddings)
            })
            
            return match_results
            
        except Exception as e:
            logger.error(f"Error in face recognition pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'probe_image': probe_image_path,
                'group_photo': group_photo_path
            }
    
    def batch_recognize(self, probe_images: List[str], group_photo_path: str) -> List[Dict]:
        """
        Batch face recognition for multiple probe images
        
        Args:
            probe_images: List of probe image paths
            group_photo_path: Path to group photo
            
        Returns:
            List of recognition results
        """
        logger.info(f"Starting batch recognition for {len(probe_images)} probe images")
        
        # Process group photo once
        cropped_faces, candidate_embeddings = self.process_group_photo(group_photo_path)
        
        if not candidate_embeddings:
            return [{
                'success': False,
                'error': 'No faces detected in group photo',
                'probe_image': probe_path,
                'group_photo': group_photo_path
            } for probe_path in probe_images]
        
        results = []
        for probe_path in probe_images:
            try:
                # Process probe image
                probe_embedding = self.process_probe_image(probe_path)
                
                # Perform matching
                match_results = self.face_matcher.match_with_details(
                    probe_embedding, candidate_embeddings
                )
                
                match_results.update({
                    'success': True,
                    'probe_image': probe_path,
                    'group_photo': group_photo_path,
                    'num_candidates': len(candidate_embeddings)
                })
                
                results.append(match_results)
                
            except Exception as e:
                logger.error(f"Error processing {probe_path}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'probe_image': probe_path,
                    'group_photo': group_photo_path
                })
        
        return results
    
    def save_cropped_faces(self, group_photo_path: str, output_dir: str) -> List[str]:
        """
        Save cropped faces from group photo
        
        Args:
            group_photo_path: Path to group photo
            output_dir: Directory to save cropped faces
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cropped_faces, _ = self.process_group_photo(group_photo_path)
        
        saved_paths = []
        for i, face in enumerate(cropped_faces):
            output_path = os.path.join(output_dir, f"face_{i+1}.jpg")
            cv2.imwrite(output_path, face)
            saved_paths.append(output_path)
            
        logger.info(f"Saved {len(saved_paths)} cropped faces to {output_dir}")
        return saved_paths