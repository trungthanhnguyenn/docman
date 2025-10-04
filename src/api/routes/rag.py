"""
RAG Pipeline Routes for DOCMAN

Provides endpoints for:
1. Server-side embedding generation
2. Reranked search
3. Full RAG query pipeline

Adapted from lumir-agentic RAG system.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends

from src.core.config import config
from src.core.models import (
    EmbeddingRequest, EmbeddingResponse,
    RerankSearchRequest, RerankSearchResponse, RerankSearchResult,
    RAGQueryRequest, RAGQueryResponse,
    SearchResult
)
from src.api.services.database_manager import DatabaseManager
from src.api.services.embedding_service import get_embedding_service
from src.api.services.reranking_service import get_reranking_service
from src.core.exceptions import DatabaseConnectionException
from src.api.dependencies import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/embedding/generate", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for texts using server-side embedding model.
    
    Default model: Qwen3-Embedding-0.6B (1024 dimensions)
    
    Args:
        request: Embedding request with texts
        
    Returns:
        EmbeddingResponse: Generated embeddings with metadata
        
    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        start_time = datetime.utcnow()
        
        # Get embedding service (singleton)
        embedding_service = get_embedding_service()
        
        # Override model if specified in request
        if request.model:
            # Reinitialize with new model (not recommended in production)
            logger.warning(f"Reinitializing embedding service with model: {request.model}")
            embedding_service = get_embedding_service(model_name=request.model, reinit=True)
        
        # Generate embeddings
        embeddings = embedding_service.embed_texts(request.texts)
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=embedding_service.model_name,
            dimension=embedding_service.embedding_dimension,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )


@router.post("/search/rerank", response_model=RerankSearchResponse)
async def rerank_search(
    request: RerankSearchRequest,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> RerankSearchResponse:
    """
    Search with optional reranking using cross-encoder.
    
    Pipeline:
    1. Generate query embedding (if not provided)
    2. Vector similarity search in Qdrant
    3. Rerank results using cross-encoder (if enabled)
    
    Args:
        request: Reranked search request
        db_manager: Database manager instance
        
    Returns:
        RerankSearchResponse: Reranked search results
        
    Raises:
        HTTPException: If search fails
    """
    try:
        start_time = datetime.utcnow()
        
        # Step 1: Generate query embedding if not provided
        query_vector = request.query_vector
        if not query_vector:
            embedding_service = get_embedding_service()
            query_vector = embedding_service.embed_text(request.query_text)
            logger.debug(f"Generated query embedding: {len(query_vector)} dimensions")
        
        search_start = datetime.utcnow()
        
        # Step 2: Vector search
        filters = request.filters or {}
        if request.session_id:
            filters["session_id"] = request.session_id
        
        result = await db_manager.get_chunks(
            query_vector=query_vector,
            filters=filters,
            limit=request.limit,
            collection_name=request.collection_name
        )
        
        search_time = int((datetime.utcnow() - search_start).total_seconds() * 1000)
        
        # Convert to dict format for reranking
        chunks = result.get("chunks", [])
        search_results = []
        for chunk in chunks:
            payload = chunk.get("payload", {})
            search_results.append({
                "chunk_id": payload.get("chunk_id", str(chunk.get("id", ""))),
                "document_id": payload.get("document_id", ""),
                "document_title": payload.get("doc_title", ""),
                "chunk_text": payload.get("chunk_content", ""),
                "similarity_score": chunk.get("score", 0.0),
                "source": "main",
                "metadata": {
                    "page_number": payload.get("page", 0),
                    "section": payload.get("section", ""),
                    **{k: v for k, v in payload.items() 
                       if k not in ["document_id", "doc_title", "chunk_content", "page", "section", "chunk_id"]}
                }
            })
        
        # Step 3: Reranking (if enabled)
        rerank_time = 0
        reranked = False
        
        if request.rerank and search_results:
            rerank_start = datetime.utcnow()
            reranking_service = get_reranking_service()
            
            rerank_top_k = request.rerank_top_k or request.limit
            rerank_response = reranking_service.rerank(
                query=request.query_text,
                results=search_results,
                top_k=rerank_top_k
            )
            
            rerank_time = rerank_response.processing_time_ms
            reranked = True
            
            # Convert rerank results to response format
            final_results = [
                RerankSearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    chunk_text=r.chunk_text,
                    original_score=r.original_score,
                    rerank_score=r.rerank_score,
                    combined_score=r.combined_score,
                    source="main",
                    metadata=r.metadata
                )
                for r in rerank_response.results
            ]
        else:
            # No reranking - use original results
            final_results = [
                RerankSearchResult(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    document_title=r["document_title"],
                    chunk_text=r["chunk_text"],
                    original_score=r["similarity_score"],
                    rerank_score=None,
                    combined_score=r["similarity_score"],
                    source=r["source"],
                    metadata=r["metadata"]
                )
                for r in search_results
            ]
        
        total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Get embedding service for model info
        embedding_service = get_embedding_service()
        reranking_service = get_reranking_service() if reranked else None
        
        return RerankSearchResponse(
            query_text=request.query_text,
            results=final_results,
            total_results=len(final_results),
            reranked=reranked,
            search_time_ms=search_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            model_info={
                "embedding_model": embedding_service.model_name,
                "embedding_dimension": str(embedding_service.embedding_dimension),
                "reranking_model": reranking_service.config.model_name if reranking_service else "none"
            }
        )
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Reranked search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> RAGQueryResponse:
    """
    Full RAG pipeline query.
    
    Pipeline:
    1. Generate query embedding
    2. Vector similarity search
    3. Rerank results (if enabled)
    4. Apply score threshold
    
    Args:
        request: RAG query request
        db_manager: Database manager instance
        
    Returns:
        RAGQueryResponse: RAG query results
        
    Raises:
        HTTPException: If query fails
    """
    try:
        start_time = datetime.utcnow()
        
        # Convert to rerank search request
        rerank_request = RerankSearchRequest(
            query_text=request.query,
            limit=request.top_k,
            rerank=request.rerank,
            rerank_top_k=request.top_k,
            filters={},
            collection_name=request.collection_name,
            session_id=request.session_id
        )
        
        # Execute reranked search
        search_response = await rerank_search(rerank_request, db_manager)
        
        # Filter by score threshold
        filtered_results = [
            r for r in search_response.results
            if r.combined_score >= request.score_threshold
        ]
        
        total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return RAGQueryResponse(
            query=request.query,
            results=filtered_results,
            total_found=len(filtered_results),
            processing_time_ms=total_time,
            metadata={
                "collection": request.collection_name or config.qdrant.default_collection_name,
                "session_id": request.session_id,
                "reranked": request.rerank,
                "score_threshold": request.score_threshold,
                "search_time_ms": search_response.search_time_ms,
                "rerank_time_ms": search_response.rerank_time_ms,
                "models": search_response.model_info
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )
