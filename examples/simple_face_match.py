#!/usr/bin/env python3
"""
Simple Face Matching Script

Usage: python simple_face_match.py probe.jpg group.jpg
"""

import sys
import os
from face_recognition_on_edge import FaceRecognitionPipeline

def main():
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python simple_face_match.py <probe_image> <group_photo>")
        print("Example: python simple_face_match.py person.jpg group.jpg")
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
    
    print(f"Analyzing: {probe_image} vs {group_photo}")
    print("=" * 50)
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        # Create pipeline with current threshold
        pipeline = FaceRecognitionPipeline(similarity_threshold=threshold)
        # Run recognition
        result = pipeline.recognize_face(probe_image, group_photo)
        
        if result['success']:
            match = "YES" if result['is_match'] else "NO"
            similarity = result['best_similarity']
            print(f"Threshold {threshold}: {match} (similarity: {similarity:.3f})")
        else:
            print(f"Threshold {threshold}: ERROR - {result.get('error', 'Unknown error')}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
