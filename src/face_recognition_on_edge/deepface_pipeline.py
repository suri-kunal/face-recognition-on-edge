"""
Simplified Face Recognition Pipeline using DeepFace
Integrates MediaPipe for face detection and DeepFace for recognition
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
from pathlib import Path

from deepface import DeepFace
from .face_detector import FaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepFacePipeline:
    """Simplified face recognition pipeline using DeepFace"""
    
    # DeepFace models and their default thresholds
    SUPPORTED_MODELS = {
        'VGG-Face': {'threshold': 0.4, 'distance_metric': 'cosine'},
        'Facenet': {'threshold': 0.4, 'distance_metric': 'cosine'},
        'Facenet512': {'threshold': 0.3, 'distance_metric': 'cosine'},
        'OpenFace': {'threshold': 0.1, 'distance_metric': 'cosine'},
        'DeepFace': {'threshold': 0.23, 'distance_metric': 'cosine'},
        'DeepID': {'threshold': 0.015, 'distance_metric': 'cosine'},
        'ArcFace': {'threshold': 0.68, 'distance_metric': 'cosine'},
        'Dlib': {'threshold': 0.07, 'distance_metric': 'cosine'},
        'SFace': {'threshold': 0.593, 'distance_metric': 'cosine'},
    }
    
    def __init__(self, 
                 models: List[str] = ['ArcFace'],
                 detection_confidence: float = 0.5,
                 target_face_size: Tuple[int, int] = (112, 112),
                 custom_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the simplified pipeline
        
        Args:
            models: List of DeepFace models to use
            detection_confidence: Confidence threshold for face detection
            target_face_size: Target size for cropped faces
            custom_thresholds: Optional custom thresholds for models
        """
        # Validate models
        for model in models:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model: {model}. Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.models = models
        self.face_detector = FaceDetector(min_detection_confidence=detection_confidence)
        self.target_face_size = target_face_size
        
        # Set thresholds (custom or default)
        self.thresholds = {}
        for model in models:
            if custom_thresholds and model in custom_thresholds:
                self.thresholds[model] = custom_thresholds[model]
            else:
                self.thresholds[model] = self.SUPPORTED_MODELS[model]['threshold']
        
        logger.info(f"DeepFace pipeline initialized with models: {models}")
        logger.info(f"Thresholds: {self.thresholds}")
    
    def detect_and_crop_faces(self, group_photo_path: str) -> List[np.ndarray]:
        """
        Detect and crop faces from group photo using MediaPipe
        
        Args:
            group_photo_path: Path to group photo
            
        Returns:
            List of cropped face images
        """
        logger.info(f"Detecting faces in: {group_photo_path}")
        
        cropped_faces = self.face_detector.detect_and_crop_faces(
            group_photo_path, self.target_face_size
        )
        
        logger.info(f"Detected {len(cropped_faces)} faces")
        return cropped_faces
    
    def compare_with_deepface(self, probe_image_path: str, cropped_face: np.ndarray, 
                             model: str) -> Dict:
        """
        Compare probe image with a cropped face using DeepFace
        
        Args:
            probe_image_path: Path to probe image
            cropped_face: Cropped face image as numpy array
            model: DeepFace model to use
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Save cropped face temporarily for DeepFace
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, cropped_face)
            
            # Use DeepFace to verify
            result = DeepFace.verify(
                img1_path=probe_image_path,
                img2_path=temp_face_path,
                model_name=model,
                distance_metric=self.SUPPORTED_MODELS[model]['distance_metric'],
                enforce_detection=False  # We already detected faces
            )
            
            # Clean up temp file
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)
            
            # Apply our threshold
            custom_threshold = self.thresholds[model]
            distance = result['distance']
            
            # For cosine distance, lower is more similar
            is_match = distance < custom_threshold
            
            return {
                'model': model,
                'distance': distance,
                'threshold': custom_threshold,
                'is_match': is_match,
                'deepface_verified': result['verified'],  # DeepFace's own decision
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in DeepFace comparison with {model}: {e}")
            return {
                'model': model,
                'distance': float('inf'),
                'threshold': self.thresholds[model],
                'is_match': False,
                'deepface_verified': False,
                'success': False,
                'error': str(e)
            }
    
    def recognize_face(self, probe_image_path: str, group_photo_path: str) -> Dict:
        """
        Complete face recognition pipeline using DeepFace
        
        Args:
            probe_image_path: Path to probe image
            group_photo_path: Path to group photo
            
        Returns:
            Dictionary with recognition results for all models
        """
        logger.info("Starting DeepFace recognition pipeline")
        
        try:
            # Check if files exist
            if not os.path.exists(probe_image_path):
                raise FileNotFoundError(f"Probe image not found: {probe_image_path}")
            if not os.path.exists(group_photo_path):
                raise FileNotFoundError(f"Group photo not found: {group_photo_path}")
            
            # Detect and crop faces from group photo
            cropped_faces = self.detect_and_crop_faces(group_photo_path)
            
            if not cropped_faces:
                return {
                    'success': False,
                    'error': 'No faces detected in group photo',
                    'probe_image': probe_image_path,
                    'group_photo': group_photo_path,
                    'num_candidates': 0,
                    'results_by_model': {}
                }
            
            # Results for each model
            results_by_model = {}
            
            for model in self.models:
                logger.info(f"Testing with model: {model}")
                
                model_results = {
                    'model': model,
                    'threshold': self.thresholds[model],
                    'face_comparisons': [],
                    'best_match': None,
                    'match_found': False
                }
                
                best_distance = float('inf')
                best_face_idx = -1
                
                # Compare probe with each cropped face
                for face_idx, cropped_face in enumerate(cropped_faces):
                    comparison = self.compare_with_deepface(
                        probe_image_path, cropped_face, model
                    )
                    comparison['face_index'] = face_idx
                    model_results['face_comparisons'].append(comparison)
                    
                    # Track best match
                    if comparison['success'] and comparison['distance'] < best_distance:
                        best_distance = comparison['distance']
                        best_face_idx = face_idx
                
                # Set best match info
                if best_face_idx >= 0:
                    best_comparison = model_results['face_comparisons'][best_face_idx]
                    model_results['best_match'] = {
                        'face_index': best_face_idx,
                        'distance': best_distance,
                        'is_match': best_comparison['is_match']
                    }
                    model_results['match_found'] = best_comparison['is_match']
                
                results_by_model[model] = model_results
            
            # Overall summary
            any_match = any(results['match_found'] for results in results_by_model.values())
            
            return {
                'success': True,
                'probe_image': probe_image_path,
                'group_photo': group_photo_path,
                'num_candidates': len(cropped_faces),
                'models_tested': self.models,
                'results_by_model': results_by_model,
                'overall_match_found': any_match
            }
            
        except Exception as e:
            logger.error(f"Error in DeepFace pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'probe_image': probe_image_path,
                'group_photo': group_photo_path,
                'results_by_model': {}
            }
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*80)
        print("DEEPFACE RECOGNITION RESULTS")
        print("="*80)
        
        if not results['success']:
            print(f"ERROR: {results.get('error', 'Unknown error')}")
            return
        
        print(f"Probe: {Path(results['probe_image']).name}")
        print(f"Group: {Path(results['group_photo']).name}")
        print(f"Candidates: {results['num_candidates']}")
        print(f"Overall Match: {'YES' if results['overall_match_found'] else 'NO'}")
        
        print(f"\nRESULTS BY MODEL:")
        print("-" * 80)
        
        for model_name, model_results in results['results_by_model'].items():
            print(f"\n{model_name} (threshold: {model_results['threshold']:.3f}):")
            
            if model_results['match_found']:
                best = model_results['best_match']
                print(f"  ✓ MATCH FOUND - Face {best['face_index']} (distance: {best['distance']:.3f})")
            else:
                print(f"  ✗ NO MATCH")
            
            # Show all face comparisons
            print(f"  All faces:")
            for comp in model_results['face_comparisons']:
                if comp['success']:
                    match_symbol = "✓" if comp['is_match'] else "✗"
                    print(f"    Face {comp['face_index']}: {match_symbol} {comp['distance']:.3f}")
                else:
                    print(f"    Face {comp['face_index']}: ERROR")
        
        print("="*80)