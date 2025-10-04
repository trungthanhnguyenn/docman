"""
Reranking Service for DOCMAN

Adapted from lumir-agentic/module/retrieval/reranker.py
Provides cross-encoder based reranking for improved search relevance.

Features:
- Cross-encoder reranking (MiniLM default)
- Score normalization
- Configurable model selection
- CPU/GPU support
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cross-encoder library
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for reranking")


@dataclass
class RerankingConfig:
    """Configuration for reranking service"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 20
    min_score_threshold: float = 0.1
    max_content_length: int = 512
    normalize_scores: bool = True


@dataclass
class RerankResult:
    """Individual reranked result"""
    chunk_id: str
    document_id: str
    document_title: str
    chunk_text: str
    original_score: float
    rerank_score: float
    combined_score: float
    metadata: Dict[str, Any]


@dataclass
class RerankingResponse:
    """Response from reranking service"""
    results: List[RerankResult]
    processing_time_ms: int
    original_count: int
    returned_count: int
    model_used: str


class RerankingService:
    """
    Cross-encoder based reranking service
    
    Uses cross-encoder models to rerank search results based on
    query-document semantic similarity.
    
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """
        Initialize reranking service
        
        Args:
            config: Reranking configuration (uses default if None)
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install: pip install sentence-transformers"
            )
        
        self.config = config or RerankingConfig()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize cross-encoder model"""
        try:
            logger.info(f"Loading cross-encoder: {self.config.model_name}")
            self.model = CrossEncoder(self.config.model_name)
            logger.info(f"✅ Cross-encoder loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.config.model_name}: {e}")
            # Fallback to smaller model
            try:
                fallback_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                logger.info(f"Attempting fallback to {fallback_model}")
                self.model = CrossEncoder(fallback_model)
                self.config.model_name = fallback_model
                logger.info(f"✅ Fallback model loaded: {fallback_model}")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise RuntimeError(f"Failed to load any cross-encoder model: {e2}")
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to max length"""
        if len(content) > self.config.max_content_length:
            return content[:self.config.max_content_length] + "..."
        return content
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores or len(scores) == 1:
            return scores
        
        import numpy as np
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            normalized = (scores_array - min_score) / (max_score - min_score)
            return normalized.tolist()
        
        return scores
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> RerankingResponse:
        """
        Rerank search results using cross-encoder
        
        Args:
            query: Search query text
            results: List of search results to rerank
                Each result should have: chunk_text, chunk_id, document_id, score, etc.
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            RerankingResponse with reranked results
        """
        start_time = time.time()
        
        if not results:
            return RerankingResponse(
                results=[],
                processing_time_ms=0,
                original_count=0,
                returned_count=0,
                model_used=self.config.model_name
            )
        
        top_k = top_k or self.config.top_k
        original_count = len(results)
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for result in results:
                content = result.get("chunk_text", "")
                content = self._truncate_content(content)
                pairs.append([query, content])
            
            # Get cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} results with cross-encoder")
            cross_scores = self.model.predict(pairs)
            
            # Normalize scores if configured
            if self.config.normalize_scores:
                cross_scores = self._normalize_scores(cross_scores.tolist())
            else:
                cross_scores = cross_scores.tolist()
            
            # Combine original and cross-encoder scores
            reranked_results = []
            for result, cross_score in zip(results, cross_scores):
                original_score = result.get("similarity_score", 0.0)
                
                # Combined score: weighted average (70% cross-encoder, 30% original)
                combined_score = 0.7 * cross_score + 0.3 * original_score
                
                reranked_results.append(RerankResult(
                    chunk_id=result.get("chunk_id", ""),
                    document_id=result.get("document_id", ""),
                    document_title=result.get("document_title", ""),
                    chunk_text=result.get("chunk_text", ""),
                    original_score=original_score,
                    rerank_score=cross_score,
                    combined_score=combined_score,
                    metadata=result.get("metadata", {})
                ))
            
            # Sort by combined score (descending)
            reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Filter by threshold and limit to top_k
            reranked_results = [
                r for r in reranked_results
                if r.combined_score >= self.config.min_score_threshold
            ][:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Reranked {original_count} -> {len(reranked_results)} results "
                f"in {processing_time}ms"
            )
            
            return RerankingResponse(
                results=reranked_results,
                processing_time_ms=processing_time,
                original_count=original_count,
                returned_count=len(reranked_results),
                model_used=self.config.model_name
            )
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results on error
            processing_time = int((time.time() - start_time) * 1000)
            
            fallback_results = [
                RerankResult(
                    chunk_id=r.get("chunk_id", ""),
                    document_id=r.get("document_id", ""),
                    document_title=r.get("document_title", ""),
                    chunk_text=r.get("chunk_text", ""),
                    original_score=r.get("similarity_score", 0.0),
                    rerank_score=r.get("similarity_score", 0.0),
                    combined_score=r.get("similarity_score", 0.0),
                    metadata=r.get("metadata", {})
                )
                for r in results[:top_k]
            ]
            
            return RerankingResponse(
                results=fallback_results,
                processing_time_ms=processing_time,
                original_count=original_count,
                returned_count=len(fallback_results),
                model_used=self.config.model_name
            )


# Singleton instance
_reranking_service_instance: Optional[RerankingService] = None


def get_reranking_service(
    config: Optional[RerankingConfig] = None,
    reinit: bool = False
) -> RerankingService:
    """
    Get or create singleton reranking service instance
    
    Args:
        config: Reranking configuration
        reinit: Force reinitialization
        
    Returns:
        RerankingService instance
    """
    global _reranking_service_instance
    
    if _reranking_service_instance is None or reinit:
        logger.info("Initializing reranking service")
        _reranking_service_instance = RerankingService(config=config)
    
    return _reranking_service_instance
