"""
Face Detection Module using MediaPipe
Handles face detection and cropping from group photos
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using MediaPipe"""
    
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1):
        """
        Initialize MediaPipe Face Detection
        
        Args:
            min_detection_confidence: Minimum confidence threshold for detection
            model_selection: 0 for short-range model, 1 for full-range model
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image and return bounding boxes
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(image_rgb)
        
        bboxes = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                bboxes.append((x, y, width, height))
                
        logger.info(f"Detected {len(bboxes)} faces")
        return bboxes
    
    def crop_faces(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                   target_size: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
        """
        Crop and resize faces from image based on bounding boxes
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes
            target_size: Target size for cropped faces (width, height)
            
        Returns:
            List of cropped and resized face images
        """
        cropped_faces = []
        
        for i, (x, y, width, height) in enumerate(bboxes):
            # Crop face with some padding
            padding = 20
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            x2_pad = min(image.shape[1], x + width + padding)
            y2_pad = min(image.shape[0], y + height + padding)
            
            face_crop = image[y_pad:y2_pad, x_pad:x2_pad]
            
            # Resize to target size
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, target_size)
                cropped_faces.append(face_resized)
            else:
                logger.warning(f"Empty face crop for bbox {i}")
                
        return cropped_faces
    
    def detect_and_crop_faces(self, image_path: str, 
                            target_size: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
        """
        Complete pipeline: detect faces and return cropped face images
        
        Args:
            image_path: Path to input image
            target_size: Target size for cropped faces
            
        Returns:
            List of cropped face images
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Detect faces
        bboxes = self.detect_faces(image)
        
        # Crop faces
        cropped_faces = self.crop_faces(image, bboxes, target_size)
        
        return cropped_faces
    
    def visualize_detections(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize face detections on image
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            
        Returns:
            Image with face detections drawn
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(image, detection)
        
        if output_path:
            cv2.imwrite(output_path, image)
            
        return image