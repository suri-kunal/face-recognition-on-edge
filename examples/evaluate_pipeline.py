"""
Evaluation script for the face recognition pipeline using LFW dataset
"""

import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os

from face_recognition_on_edge import FaceRecognitionPipeline, LFWDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Evaluator for face recognition pipeline"""
    
    def __init__(self, pipeline: FaceRecognitionPipeline):
        """
        Initialize evaluator
        
        Args:
            pipeline: Face recognition pipeline to evaluate
        """
        self.pipeline = pipeline
        self.results = []
    
    def evaluate_pairs(self, test_pairs: List[Tuple[str, str, bool]]) -> dict:
        """
        Evaluate pipeline on pairs of images
        
        Args:
            test_pairs: List of (image1_path, image2_path, is_same_person)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {len(test_pairs)} pairs...")
        
        similarities = []
        labels = []
        correct_predictions = 0
        
        for i, (img1_path, img2_path, is_same) in enumerate(test_pairs):
            try:
                # Extract features from both images
                emb1 = self.pipeline.feature_extractor.extract_from_path(img1_path)
                emb2 = self.pipeline.feature_extractor.extract_from_path(img2_path)
                
                # Compute similarity
                similarity = self.pipeline.face_matcher.cosine_similarity(emb1, emb2)
                
                # Predict using current threshold
                prediction = similarity > self.pipeline.face_matcher.similarity_threshold
                
                similarities.append(similarity)
                labels.append(is_same)
                
                if prediction == is_same:
                    correct_predictions += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_pairs)} pairs")
                    
            except Exception as e:
                logger.error(f"Error processing pair {i}: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / len(similarities) if similarities else 0
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'num_pairs': len(similarities),
            'similarities': similarities,
            'labels': labels,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        self.results.append(results)
        return results
    
    def plot_roc_curve(self, results: dict, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            results: Evaluation results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], 
                label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Face Recognition Pipeline')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: dict, save_path: str):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        # Remove numpy arrays for JSON serialization
        results_copy = results.copy()
        results_copy.pop('similarities', None)
        results_copy.pop('labels', None)
        
        with open(save_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")


def evaluate_with_lfw():
    """Evaluate pipeline using LFW dataset"""
    logger.info("=== LFW Evaluation ===")
    
    # Setup LFW dataset
    lfw = LFWDataset("data/lfw")
    if not lfw.setup_dataset():
        logger.error("Failed to setup LFW dataset")
        return
    
    # Create test pairs
    test_pairs = lfw.create_test_pairs(num_pairs=200)
    if not test_pairs:
        logger.error("Failed to create test pairs")
        return
    
    # Initialize pipeline with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results_summary = []
    
    for threshold in thresholds:
        logger.info(f"Evaluating with threshold: {threshold}")
        
        pipeline = FaceRecognitionPipeline(similarity_threshold=threshold)
        evaluator = PipelineEvaluator(pipeline)
        
        results = evaluator.evaluate_pairs(test_pairs)
        results['threshold'] = threshold
        results_summary.append(results)
        
        logger.info(f"Threshold {threshold}: Accuracy = {results['accuracy']:.3f}, "
                   f"AUC = {results['roc_auc']:.3f}")
    
    # Find best threshold
    best_result = max(results_summary, key=lambda x: x['accuracy'])
    logger.info(f"Best threshold: {best_result['threshold']} "
               f"(Accuracy: {best_result['accuracy']:.3f})")
    
    # Plot ROC curve for best result
    evaluator = PipelineEvaluator(FaceRecognitionPipeline())
    evaluator.plot_roc_curve(best_result, "data/roc_curve.png")
    
    # Save detailed results
    evaluator.save_results(best_result, "data/evaluation_results.json")
    
    return results_summary


def evaluate_group_photo_recognition():
    """Evaluate group photo recognition capability"""
    logger.info("=== Group Photo Recognition Evaluation ===")
    
    # Setup LFW dataset
    lfw = LFWDataset("data/lfw")
    if not lfw.setup_dataset():
        return
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(similarity_threshold=0.5)
    
    # Test with different group sizes
    group_sizes = [2, 3, 5, 7, 10]
    results = []
    
    for group_size in group_sizes:
        logger.info(f"Testing group size: {group_size}")
        
        num_tests = 10
        correct_matches = 0
        
        for test_idx in range(num_tests):
            try:
                # Create group photo
                group_photo_path, individual_images = lfw.create_group_photo_simulation(
                    num_faces=group_size,
                    output_path=f"data/eval_group_{group_size}_{test_idx}.jpg"
                )
                
                # Test with first individual
                if individual_images:
                    probe_image = individual_images[0]
                    
                    recognition_result = pipeline.recognize_face(probe_image, group_photo_path)
                    
                    if recognition_result['success'] and recognition_result['is_match']:
                        correct_matches += 1
                
                # Clean up
                if os.path.exists(group_photo_path):
                    os.remove(group_photo_path)
                    
            except Exception as e:
                logger.error(f"Error in group test {test_idx}: {e}")
                continue
        
        accuracy = correct_matches / num_tests
        results.append({
            'group_size': group_size,
            'accuracy': accuracy,
            'correct_matches': correct_matches,
            'total_tests': num_tests
        })
        
        logger.info(f"Group size {group_size}: {correct_matches}/{num_tests} "
                   f"correct ({accuracy:.2%})")
    
    return results


def main():
    """Run evaluation"""
    print("Face Recognition Pipeline Evaluation")
    print("====================================")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run evaluations
    lfw_results = evaluate_with_lfw()
    group_results = evaluate_group_photo_recognition()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if lfw_results:
        print("\nLFW Pair Evaluation:")
        for result in lfw_results:
            print(f"  Threshold {result['threshold']}: "
                  f"Accuracy = {result['accuracy']:.3f}, "
                  f"AUC = {result['roc_auc']:.3f}")
    
    if group_results:
        print("\nGroup Photo Recognition:")
        for result in group_results:
            print(f"  Group size {result['group_size']}: "
                  f"Accuracy = {result['accuracy']:.2%}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
