"""
LFW Dataset utilities
Handles downloading and processing of the Labeled Faces in the Wild dataset
"""

import os
import requests
import tarfile
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import random
from PIL import Image
import numpy as np
import pdb
import kagglehub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LFWDataset:
    """Handler for LFW dataset"""
    
    def __init__(self, data_dir: str = "data/lfw"):
        """
        Initialize LFW dataset handler
        
        Args:
            data_dir: Directory to store LFW data
        """
        os.environ["KAGGLEHUB_CACHE"] = data_dir
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.lfw_url = "jessicali9530/lfw-dataset"
        self.lfw_dir = \
        Path("data/lfw/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled")
        
    def download_lfw(self) -> bool:
        """
        Download LFW dataset if not already present
        
        Returns:
            True if successful, False otherwise
        """
        if self.data_dir.exists() and len(list(self.data_dir.iterdir())) > 0:
            logger.info("LFW dataset already exists")
            return True
            
        try:
            logger.info("Downloading LFW dataset...")
            # Download latest version
            path = kagglehub.dataset_download(self.lfw_url,force_download=True)

            print("Path to dataset files:", path)
            logger.info("LFW dataset downloaded successfully")

            return True
            
        except Exception as e:
            logger.error(f"Error downloading LFW dataset: {e}")
            return False
    
    def get_person_images(self, person_name: str) -> List[Path]:
        """
        Get all images for a specific person
        
        Args:
            person_name: Name of the person
            
        Returns:
            List of image paths for the person
        """
        person_dir = self.lfw_dir / person_name
        if not person_dir.exists():
            return []
            
        image_files = list(person_dir.glob("*.jpg"))
        return sorted(image_files)
    
    def get_all_persons(self) -> List[str]:
        """
        Get list of all person names in the dataset
        
        Returns:
            List of person names
        """
        if not self.lfw_dir.exists():
            return []
        persons = [d.name for d in self.lfw_dir.iterdir() if d.is_dir()]
        
        
        return sorted(persons)
    
    def create_test_pairs(self, num_pairs: int = 100) -> List[Tuple[str, str, bool]]:
        """
        Create test pairs for face recognition evaluation
        
        Args:
            num_pairs: Number of pairs to create
            
        Returns:
            List of tuples (image1_path, image2_path, is_same_person)
        """
        persons = self.get_all_persons()
        if len(persons) < 2:
            logger.error("Not enough persons in dataset")
            return []
        
        pairs = []
        
        # Create positive pairs (same person)
        for _ in range(num_pairs // 2):
            person = random.choice(persons)
            person_images = self.get_person_images(person)
            
            if len(person_images) >= 2:
                img1, img2 = random.sample(person_images, 2)
                pairs.append((str(img1), str(img2), True))
        
        # Create negative pairs (different persons)
        for _ in range(num_pairs // 2):
            person1, person2 = random.sample(persons, 2)
            images1 = self.get_person_images(person1)
            images2 = self.get_person_images(person2)
            
            if images1 and images2:
                img1 = random.choice(images1)
                img2 = random.choice(images2)
                pairs.append((str(img1), str(img2), False))
        
        random.shuffle(pairs)
        logger.info(f"Created {len(pairs)} test pairs")
        return pairs
    
    def create_group_photo_simulation(self, num_faces: int = 5, output_path: str = None) -> Tuple[str, List[str]]:
        """
        Create a simulated group photo by combining multiple face images
        
        Args:
            num_faces: Number of faces to include
            output_path: Path to save the group photo
            
        Returns:
            Tuple of (group_photo_path, list_of_individual_image_paths)
        """
        persons = self.get_all_persons()
        if len(persons) < num_faces:
            raise ValueError(f"Not enough persons in dataset. Need {num_faces}, have {len(persons)}")
        
        selected_persons = random.sample(persons, num_faces)
        individual_images = []
        face_images = []
        
        for person in selected_persons:
            person_images = self.get_person_images(person)
            if person_images:
                selected_image = random.choice(person_images)
                individual_images.append(str(selected_image))
                
                # Load and resize image
                img = Image.open(selected_image)
                img = img.resize((112, 112))
                face_images.append(np.array(img))
        
        # Create a simple grid layout for the group photo
        if len(face_images) == 0:
            raise ValueError("No valid face images found")
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(len(face_images))))
        rows = int(np.ceil(len(face_images) / cols))
        
        # Create group photo
        group_height = rows * 112
        group_width = cols * 112
        group_photo = np.zeros((group_height, group_width, 3), dtype=np.uint8)
        
        for i, face_img in enumerate(face_images):
            row = i // cols
            col = i % cols
            
            start_y = row * 112
            start_x = col * 112
            
            if len(face_img.shape) == 3:
                group_photo[start_y:start_y+112, start_x:start_x+112] = face_img
            else:
                # Convert grayscale to RGB
                face_rgb = np.stack([face_img] * 3, axis=-1)
                group_photo[start_y:start_y+112, start_x:start_x+112] = face_rgb
        
        # Save group photo
        if output_path is None:
            output_path = str(self.data_dir / "simulated_group_photo.jpg")
        
        group_pil = Image.fromarray(group_photo)
        group_pil.save(output_path)
        
        logger.info(f"Created simulated group photo with {len(face_images)} faces: {output_path}")
        return output_path, individual_images
    
    def setup_dataset(self) -> bool:
        """
        Setup complete LFW dataset
        
        Returns:
            True if successful, False otherwise
        """
        success = True
        success &= self.download_lfw()
        # success &= self.download_pairs_file()
        
        if success:
            logger.info("LFW dataset setup completed successfully")
        else:
            logger.error("Failed to setup LFW dataset")
            
        return success