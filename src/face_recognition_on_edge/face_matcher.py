"""
Face Matching Module
Handles comparison of face embeddings and similarity computation
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceMatcher:
    """Face matcher using cosine similarity"""
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize face matcher
        
        Args:
            similarity_threshold: Threshold for considering a match
        """
        self.similarity_threshold = similarity_threshold
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Reshape to 2D arrays for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def compute_similarities(self, probe_embedding: np.ndarray, 
                           candidate_embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute similarities between probe and all candidates
        
        Args:
            probe_embedding: Embedding of the probe image
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for candidate_embedding in candidate_embeddings:
            similarity = self.cosine_similarity(probe_embedding, candidate_embedding)
            similarities.append(similarity)
            
        return similarities
    
    def find_best_match(self, probe_embedding: np.ndarray, 
                       candidate_embeddings: List[np.ndarray]) -> Tuple[int, float, bool]:
        """
        Find the best matching candidate
        
        Args:
            probe_embedding: Embedding of the probe image
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            Tuple of (best_match_index, best_similarity_score, is_match)
        """
        if not candidate_embeddings:
            return -1, 0.0, False
            
        similarities = self.compute_similarities(probe_embedding, candidate_embeddings)
        
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]
        is_match = best_similarity > self.similarity_threshold
        
        logger.info(f"Best match: index {best_match_index}, "
                   f"similarity {best_similarity:.3f}, "
                   f"is_match: {is_match}")
        
        return best_match_index, best_similarity, is_match
    
    def match_with_details(self, probe_embedding: np.ndarray, 
                          candidate_embeddings: List[np.ndarray]) -> dict:
        """
        Perform matching with detailed results
        
        Args:
            probe_embedding: Embedding of the probe image
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            Dictionary with detailed matching results
        """
        similarities = self.compute_similarities(probe_embedding, candidate_embeddings)
        
        if not similarities:
            return {
                'best_match_index': -1,
                'best_similarity': 0.0,
                'is_match': False,
                'all_similarities': [],
                'threshold': self.similarity_threshold,
                'num_candidates': 0
            }
        
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]
        is_match = best_similarity > self.similarity_threshold
        
        return {
            'best_match_index': best_match_index,
            'best_similarity': best_similarity,
            'is_match': is_match,
            'all_similarities': similarities,
            'threshold': self.similarity_threshold,
            'num_candidates': len(candidate_embeddings)
        }
    
    def set_threshold(self, threshold: float):
        """Update similarity threshold"""
        self.similarity_threshold = threshold
        logger.info(f"Updated similarity threshold to {threshold}")