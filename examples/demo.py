"""
Demo script showing how to use the face recognition pipeline
"""

import cv2
import numpy as np
from pathlib import Path
import logging

from face_recognition_on_edge import FaceRecognitionPipeline, LFWDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_recognition():
    """Demo basic face recognition functionality"""
    logger.info("=== Basic Face Recognition Demo ===")
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(
        similarity_threshold=0.5,
        detection_confidence=0.5
    )
    
    # Setup LFW dataset
    lfw = LFWDataset()
    if not lfw.setup_dataset():
        logger.error("Failed to setup LFW dataset")
        return
    
    # Create a simulated group photo
    try:
        group_photo_path, individual_images = lfw.create_group_photo_simulation(
            num_faces=5, output_path="data/demo_group_photo.jpg"
        )
        logger.info(f"Created group photo: {group_photo_path}")
        logger.info(f"Individual images: {individual_images}")
        
        # Test recognition with one of the individuals
        if individual_images:
            probe_image = individual_images[0]
            logger.info(f"Using probe image: {probe_image}")
            
            # Run recognition
            results = pipeline.recognize_face(probe_image, group_photo_path)
            
            # Print results
            print("\n" + "="*50)
            print("RECOGNITION RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            if results['success']:
                print(f"Number of candidates: {results['num_candidates']}")
                print(f"Best match index: {results['best_match_index']}")
                print(f"Best similarity: {results['best_similarity']:.3f}")
                print(f"Is match: {results['is_match']}")
                print(f"Threshold: {results['threshold']}")
                print(f"All similarities: {[f'{s:.3f}' for s in results['all_similarities']]}")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")
            print("="*50)
            
    except Exception as e:
        logger.error(f"Error in demo: {e}")


def demo_batch_recognition():
    """Demo batch recognition functionality"""
    logger.info("=== Batch Recognition Demo ===")
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(similarity_threshold=0.4)
    
    # Setup LFW dataset
    lfw = LFWDataset("data/lfw")
    if not lfw.setup_dataset():
        logger.error("Failed to setup LFW dataset")
        return
    
    try:
        # Create group photo
        group_photo_path, individual_images = lfw.create_group_photo_simulation(
            num_faces=3, output_path="data/batch_demo_group_photo.jpg"
        )
        
        # Test with multiple probe images
        probe_images = individual_images[:2] if len(individual_images) >= 2 else individual_images
        
        # Run batch recognition
        results = pipeline.batch_recognize(probe_images, group_photo_path)
        
        # Print results
        print("\n" + "="*50)
        print("BATCH RECOGNITION RESULTS")
        print("="*50)
        for i, result in enumerate(results):
            print(f"\nProbe {i+1}: {Path(result['probe_image']).name}")
            if result['success']:
                print(f"  Best match index: {result['best_match_index']}")
                print(f"  Best similarity: {result['best_similarity']:.3f}")
                print(f"  Is match: {result['is_match']}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in batch demo: {e}")


def demo_threshold_tuning():
    """Demo threshold tuning"""
    logger.info("=== Threshold Tuning Demo ===")
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Setup LFW dataset
    lfw = LFWDataset("data/lfw")
    if not lfw.setup_dataset():
        return
    
    try:
        # Create group photo
        group_photo_path, individual_images = lfw.create_group_photo_simulation(
            num_faces=4, output_path="data/threshold_demo_group_photo.jpg"
        )
        
        if individual_images:
            probe_image = individual_images[0]
            
            # Test different thresholds
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            
            print("\n" + "="*60)
            print("THRESHOLD TUNING RESULTS")
            print("="*60)
            print(f"{'Threshold':<10} {'Best Sim':<10} {'Is Match':<10} {'Match Index':<12}")
            print("-" * 60)
            
            for threshold in thresholds:
                pipeline.face_matcher.set_threshold(threshold)
                results = pipeline.recognize_face(probe_image, group_photo_path)
                
                if results['success']:
                    print(f"{threshold:<10.1f} {results['best_similarity']:<10.3f} "
                          f"{str(results['is_match']):<10} {results['best_match_index']:<12}")
                else:
                    print(f"{threshold:<10.1f} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
            
            print("="*60)
            
    except Exception as e:
        logger.error(f"Error in threshold demo: {e}")


def main():
    """Run all demos"""
    print("Face Recognition Pipeline Demo")
    print("==============================")
    
    # Create data directory
    import os
    os.makedirs("data", exist_ok=True)
    
    # Run demos
    demo_basic_recognition()
    demo_batch_recognition()
    demo_threshold_tuning()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
