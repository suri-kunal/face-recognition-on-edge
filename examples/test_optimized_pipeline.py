#!/usr/bin/env python3
"""
Test script for optimized DeepFace pipeline with model caching
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_recognition_on_edge.deepface_pipeline import DeepFacePipeline

def test_optimized_pipeline():
    """Test the optimized pipeline with model preloading"""
    
    # Test with preloading enabled (default)
    print("=" * 60)
    print("TESTING OPTIMIZED PIPELINE (with model preloading)")
    print("=" * 60)
    
    pipeline = DeepFacePipeline(
        models=['ArcFace', 'Facenet', 'VGG-Face'],
        preload_models=True  # Enable model caching
    )
    
    # Test recognition
    probe_image = "Namita Passport Photo.jpg"
    group_photo = "IMG_20250820_201407.jpg"
    
    if os.path.exists(probe_image) and os.path.exists(group_photo):
        print(f"\nRunning face recognition...")
        print(f"Probe: {probe_image}")
        print(f"Group: {group_photo}")
        
        results = pipeline.recognize_face(probe_image, group_photo)
        
        if results['success']:
            print(f"\nResults:")
            print(f"Overall Match: {'YES' if results['overall_match_found'] else 'NO'}")
            print(f"Candidates: {results['num_candidates']}")
            
            for model, result in results['results_by_model'].items():
                match_status = "✓ MATCH FOUND" if result['match_found'] else "✗ NO MATCH"
                best_distance = result['best_match']['distance'] if result['best_match'] else 'N/A'
                print(f"{model}: {match_status} (best distance: {best_distance})")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
    else:
        print(f"Test images not found:")
        print(f"  - {probe_image}: {'✓' if os.path.exists(probe_image) else '✗'}")
        print(f"  - {group_photo}: {'✓' if os.path.exists(group_photo) else '✗'}")

if __name__ == "__main__":
    test_optimized_pipeline()
