"""
Simple DeepFace Recognition Script

Usage: python simple_deepface_match.py probe.jpg group.jpg
"""

import sys
import os
from face_recognition_on_edge.deepface_pipeline import DeepFacePipeline

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_deepface_match.py <probe_image> <group_photo>")
        print("Example: python simple_deepface_match.py person.jpg group.jpg")
        sys.exit(1)
    
    probe_image = sys.argv[1]
    group_photo = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(probe_image):
        print(f"Error: Probe image '{probe_image}' not found!")
        sys.exit(1)
    
    if not os.path.exists(group_photo):
        print(f"Error: Group photo '{group_photo}' not found!")
        sys.exit(1)
    
    # Initialize pipeline with multiple models
    pipeline = DeepFacePipeline(
        models=['ArcFace', 'Facenet', 'VGG-Face'],  # Test 3 different models
        detection_confidence=0.5
    )
    
    # Run recognition
    results = pipeline.recognize_face(probe_image, group_photo)
    
    # Print results
    pipeline.print_results(results)

if __name__ == "__main__":
    main()