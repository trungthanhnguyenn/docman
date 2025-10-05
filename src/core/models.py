# src/core/models.py
"""
Centralized data models for the Document Management System.
All Pydantic BaseModel schemas are consolidated here to maintain consistency 
and support the 4 core features:
1. Session management by CRUD
2. Document management by CRUD  
3. Chunks management by CRUD
4. Metrics and logging

This file contains all request/response models, data structures, and validation logic.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID


# =============================================
# SESSION MODELS
# =============================================

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier who owns the session")
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    status: str = Field(..., description="Session status (active, expired, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    temp_collection_name: Optional[str] = Field(None, description="Temporary collection name for this session")


class SessionCreateRequest(BaseModel):
    """Session creation request"""
    user_id: str = Field(..., description="User identifier")
    expires_in_hours: int = Field(24, ge=1, le=168, description="Session duration in hours (1-168)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Initial session metadata")
    temp_collection_name: Optional[str] = Field(None, description="Custom temporary collection name")


class SessionUpdateRequest(BaseModel):
    """Session update request"""
    status: Optional[str] = Field(None, description="New session status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    temp_collection_name: Optional[str] = Field(None, description="Updated temporary collection name")
    extend_hours: Optional[int] = Field(None, ge=1, le=168, description="Extend expiration by X hours")


class AdminStatsResponse(BaseModel):
    """Administrative statistics response"""
    total_sessions: int = Field(..., ge=0, description="Total number of sessions")
    active_sessions: int = Field(..., ge=0, description="Number of active sessions")
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    total_searches: int = Field(..., ge=0, description="Total number of searches performed")
    system_uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")
    database_status: Dict[str, bool] = Field(..., description="Database connection status")


# =============================================
# DOCUMENT MODELS
# =============================================

class Document(BaseModel):
    """Document response model"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    session_id: str = Field(..., description="Associated session ID")
    user_id: str = Field(..., description="User who uploaded the document")
    status: str = Field(..., description="Processing status")


# Document payload models aligned with database implementations
class DocumentPayload(BaseModel):
    """Document payload structure for Qdrant chunks"""
    document_id: str = Field(..., description="Unique document identifier")
    doc_title: str = Field("", description="Document title")
    page: int = Field(0, description="Page number")
    chunk_content: str = Field(..., description="Chunk text content")
    file_url: str = Field("", description="Document file URL")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()
    
    @validator('chunk_content')
    def validate_chunk_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v.strip()


class DocumentMetadata(BaseModel):
    """Document metadata structure for PostgreSQL storage"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    file_hash: Optional[str] = Field(None, description="File hash for duplicate detection")
    chunks_count: int = Field(0, ge=0, description="Number of chunks")
    processing_status: str = Field("uploaded", description="Processing status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., ge=1, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError('Filename cannot be empty')
        
        # Check file extension
        allowed_extensions = ['pdf', 'docx', 'txt', 'md', 'rtf']
        file_extension = v.lower().split('.')[-1] if '.' in v else ''
        if file_extension not in allowed_extensions:
            raise ValueError(f'File type not allowed. Allowed types: {allowed_extensions}')
        
        return v.strip()
    
    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 50 * 1024 * 1024  # 50MB
        if v > max_size:
            raise ValueError(f'File size {v} exceeds maximum limit of {max_size} bytes')
        return v


class DocumentUploadResponse(BaseModel):
    """Document upload response following FR002 format"""
    documents: List[DocumentMetadata] = Field(default_factory=list, description="Successfully uploaded documents")
    total_processed: int = Field(0, description="Total documents processed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    failed_uploads: Optional[List[Dict[str, str]]] = Field(None, description="Failed upload details")


# =============================================
# CHUNK MODELS  
# =============================================

class ChunkMetadata(BaseModel):
    """Chunk metadata for upload to Qdrant"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    document_title: str = Field(..., description="Document title/filename")
    chunk_text: str = Field(..., description="Chunk text content")
    vector: List[float] = Field(..., description="Embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    page_number: Optional[int] = Field(None, description="Page number")
    section: Optional[str] = Field(None, description="Document section")


class ChunkInsertRequest(BaseModel):
    """Chunk insertion request"""
    vector: List[float] = Field(..., description="Embedding vector")
    payload: DocumentPayload = Field(..., description="Chunk payload")
    id: Optional[str] = Field(None, description="Chunk ID (generated if not provided)")


class ChunkInsertResponse(BaseModel):
    """Chunk insertion response"""
    status: str = Field(..., description="Response status")
    points_processed: int = Field(0, description="Number of points processed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    failed_points: Optional[List[Dict[str, Any]]] = Field(None, description="Failed point details")


class ChunkUploadRequest(BaseModel):
    """Request model for uploading chunks"""
    chunks: List[ChunkMetadata] = Field(..., description="List of chunks to upload")
    collection_name: Optional[str] = Field(None, description="Optional collection name for Qdrant storage")


class ChunkUploadResponse(BaseModel):
    """Response model for chunk upload"""
    status: str = Field(..., description="Upload status")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    failed_chunks: Optional[List[Dict[str, Any]]] = Field(None, description="Failed chunk details")


class ChunkUpdateRequest(BaseModel):
    """Request model for updating chunks"""
    chunk_id: str = Field(..., description="Chunk ID to update")
    chunk_text: Optional[str] = Field(None, description="Updated chunk text")
    vector: Optional[List[float]] = Field(None, description="Updated embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class ChunkBatchUpdateRequest(BaseModel):
    """Request model for batch updating chunks"""
    updates: List[ChunkUpdateRequest] = Field(..., description="List of chunk updates")
    collection_name: Optional[str] = Field(None, description="Optional collection name for Qdrant storage")


class ChunkDeleteRequest(BaseModel):
    """Request model for deleting chunks"""
    chunk_ids: List[str] = Field(..., description="List of chunk IDs to delete")
    collection_name: Optional[str] = Field(None, description="Optional collection name for Qdrant storage")


class ChunkOperationResponse(BaseModel):
    """Response model for chunk operations"""
    status: str = Field(..., description="Operation status")
    chunks_affected: int = Field(..., description="Number of chunks affected")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details if any")


# =============================================
# SEARCH MODELS  
# =============================================

class SearchRequest(BaseModel):
    """Search request model"""
    query_vector: List[float] = Field(..., description="Query embedding vector")
    limit: Optional[int] = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional search filters")
    collection_name: Optional[str] = Field(None, description="Optional collection name for Qdrant search")


class SearchResult(BaseModel):
    """Individual search result"""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    document_title: str = Field(..., description="Document title")
    chunk_text: str = Field(..., description="Relevant text excerpt")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    source: str = Field(..., description="Source type (main|temp)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Search response model"""
    query_vector: List[float] = Field(..., description="Original query vector")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results")
    search_time_ms: int = Field(..., ge=0, description="Search time in milliseconds")


# =============================================
# HEALTH & MONITORING MODELS
# =============================================

class HealthStatus(BaseModel):
    """Health status response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")
    version: str = Field("1.0.0", description="System version")


class DatabaseHealth(BaseModel):
    """Database health status model"""
    minio: bool = Field(..., description="MinIO connection status")
    qdrant: bool = Field(..., description="Qdrant connection status")
    postgres: bool = Field(..., description="PostgreSQL connection status")
    overall: bool = Field(..., description="Overall database health status")


class DetailedHealthResponse(BaseModel):
    """Detailed health response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    databases: DatabaseHealth = Field(..., description="Database health information")
    system: Dict[str, Any] = Field(default_factory=dict, description="System information")
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


# =============================================
# GENERIC OPERATION MODELS
# =============================================

class DatabaseOperationResponse(BaseModel):
    """Generic database operation response"""
    status: str = Field(..., description="Operation status (success/failed)")
    message: Optional[str] = Field(None, description="Operation message")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")


# =============================================
# PAGINATION & UTILITY MODELS
# =============================================

class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=1000, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    data: List[Any] = Field(..., description="Response data")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


# =============================================
# RAG PIPELINE MODELS
# =============================================

class EmbeddingRequest(BaseModel):
    """Request for server-side embedding generation"""
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(None, description="Embedding model to use (default: Qwen3-0.6B)")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one text is required')
        # Filter empty strings
        v = [t.strip() for t in v if t and t.strip()]
        if not v:
            raise ValueError('All texts are empty after stripping')
        return v


class EmbeddingResponse(BaseModel):
    """Response from embedding generation"""
    embeddings: List[List[float]] = Field(..., description="Generated embedding vectors")
    model: str = Field(..., description="Model used for embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")


class ChunkUploadWithEmbeddingRequest(BaseModel):
    """Chunk upload request with optional server-side embedding"""
    chunks: List[Dict[str, Any]] = Field(..., description="Chunks with text content")
    session_id: str = Field(..., description="Session identifier")
    collection_name: Optional[str] = Field(None, description="Optional collection name")
    generate_embeddings: bool = Field(True, description="Generate embeddings server-side")
    embedding_model: Optional[str] = Field(None, description="Embedding model (default: Qwen3-0.6B)")


class RerankSearchRequest(BaseModel):
    """Search request with reranking support"""
    query_text: str = Field(..., description="Search query text")
    query_vector: Optional[List[float]] = Field(None, description="Pre-computed query vector (generated if not provided)")
    limit: int = Field(10, ge=1, le=100, description="Number of initial results to retrieve")
    rerank: bool = Field(True, description="Apply reranking to results")
    rerank_top_k: Optional[int] = Field(None, description="Number of results after reranking (default: limit)")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    collection_name: Optional[str] = Field(None, description="Collection to search")
    session_id: Optional[str] = Field(None, description="Session context")


class RerankSearchResult(BaseModel):
    """Search result with reranking scores"""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    document_title: str = Field(..., description="Document title")
    chunk_text: str = Field(..., description="Chunk content")
    original_score: float = Field(..., ge=0.0, le=1.0, description="Original vector similarity score")
    rerank_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cross-encoder rerank score")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Final combined score")
    source: str = Field(..., description="Source collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RerankSearchResponse(BaseModel):
    """Response from reranked search"""
    query_text: str = Field(..., description="Original query text")
    results: List[RerankSearchResult] = Field(..., description="Reranked search results")
    total_results: int = Field(..., ge=0, description="Total results found")
    reranked: bool = Field(..., description="Whether reranking was applied")
    search_time_ms: int = Field(..., ge=0, description="Vector search time")
    rerank_time_ms: int = Field(0, ge=0, description="Reranking time")
    total_time_ms: int = Field(..., ge=0, description="Total processing time")
    model_info: Dict[str, str] = Field(default_factory=dict, description="Models used (embedding, reranking)")


class RAGQueryRequest(BaseModel):
    """Full RAG pipeline query request"""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session context")
    collection_name: Optional[str] = Field(None, description="Collection to search")
    top_k: int = Field(10, ge=1, le=50, description="Number of chunks to retrieve")
    rerank: bool = Field(True, description="Apply reranking")
    score_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_metadata: bool = Field(True, description="Include full metadata in response")


class RAGQueryResponse(BaseModel):
    """Full RAG pipeline query response"""
    query: str = Field(..., description="Original query")
    results: List[RerankSearchResult] = Field(..., description="Retrieved and reranked results")
    total_found: int = Field(..., ge=0, description="Total results found")
    processing_time_ms: int = Field(..., ge=0, description="Total processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata and metrics")


# =============================================
# HEALTH & MONITORING MODELS
# =============================================

# =============================================
# CHUNKING V2 MODELS
# =============================================

class ChunkingRequest(BaseModel):
    """Request model for document chunking v2"""
    content: str = Field(..., description="Document content to chunk")
    chunk_mode: str = Field(
        "auto",
        description="Chunking strategy: auto, header_based, qa_based, semantic, size_based, advanced_semantic"
    )
    filename: Optional[str] = Field(None, description="Optional filename for context")
    document_id: Optional[str] = Field(None, description="Optional document ID")
    chunk_size: Optional[int] = Field(1000, ge=100, le=5000, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000, description="Chunk overlap in characters")
    min_chunk_size: Optional[int] = Field(50, ge=10, le=500, description="Minimum chunk size")
    
    @validator('chunk_mode')
    def validate_chunk_mode(cls, v):
        allowed_modes = ["auto", "header_based", "qa_based", "semantic", "size_based", "advanced_semantic"]
        if v not in allowed_modes:
            raise ValueError(f'chunk_mode must be one of: {", ".join(allowed_modes)}')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class ChunkV2(BaseModel):
    """Individual chunk result from v2 chunking"""
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Chunk index in document")
    start_char: int = Field(..., ge=0, description="Start character position")
    end_char: int = Field(..., ge=0, description="End character position")
    chunk_type: str = Field(..., description="Chunk type (header, content, qa_pair, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    token_count: int = Field(..., ge=0, description="Estimated token count")
    language: str = Field("vi", description="Detected language")


class ChunkingV2Response(BaseModel):
    """Response model for document chunking v2"""
    chunks: List[ChunkV2] = Field(..., description="List of generated chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    strategy_used: str = Field(..., description="Chunking strategy that was used")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document analysis metadata")


class DocumentChunkingRequest(BaseModel):
    """Request model for chunking a document by ID"""
    document_id: str = Field(..., description="Document ID to chunk")
    chunk_mode: str = Field("auto", description="Chunking strategy")
    chunk_size: Optional[int] = Field(1000, ge=100, le=5000, description="Chunk size")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000, description="Chunk overlap")
    
    @validator('chunk_mode')
    def validate_chunk_mode(cls, v):
        allowed_modes = ["auto", "header_based", "qa_based", "semantic", "size_based", "advanced_semantic"]
        if v not in allowed_modes:
            raise ValueError(f'chunk_mode must be one of: {", ".join(allowed_modes)}')
        return v


# =============================================
# DIRECT UPSERT MODELS
# =============================================

class VectorPoint(BaseModel):
    """Single vector point for upsert"""
    id: Optional[str] = Field(None, description="Point ID (auto-generated if not provided)")
    vector: List[float] = Field(..., description="Embedding vector")
    payload: Dict[str, Any] = Field(..., description="Point payload/metadata")
    
    @validator('vector')
    def validate_vector(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Vector cannot be empty')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('Vector must contain only numeric values')
        return v
    
    @validator('payload')
    def validate_payload(cls, v):
        if not v:
            raise ValueError('Payload cannot be empty')
        # Ensure basic required fields
        if 'chunk_content' not in v and 'content' not in v and 'text' not in v:
            raise ValueError('Payload must contain at least one of: chunk_content, content, or text')
        return v


class DirectUpsertRequest(BaseModel):
    """Direct upsert request to Qdrant (no embedding generation)"""
    points: List[VectorPoint] = Field(..., description="List of vector points to upsert")
    collection_name: Optional[str] = Field(None, description="Target collection name (default: document_chunks)")
    vector_size: Optional[int] = Field(None, description="Vector dimension (for collection creation if needed)")
    distance_metric: Optional[str] = Field("cosine", description="Distance metric: cosine, euclidean, dot")
    create_collection: bool = Field(False, description="Create collection if not exists")
    batch_size: int = Field(100, ge=1, le=1000, description="Batch size for upsert operations")
    
    @validator('points')
    def validate_points(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one point is required')
        return v
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        allowed = ['cosine', 'euclidean', 'dot', 'manhattan']
        if v.lower() not in allowed:
            raise ValueError(f'Distance metric must be one of: {", ".join(allowed)}')
        return v.lower()


class DirectUpsertResponse(BaseModel):
    """Response from direct upsert operation"""
    status: str = Field(..., description="Operation status: success, partial_failure, error")
    points_processed: int = Field(..., ge=0, description="Number of points successfully upserted")
    points_failed: int = Field(0, ge=0, description="Number of points that failed")
    collection_name: str = Field(..., description="Target collection name")
    vector_dimension: Optional[int] = Field(None, description="Vector dimension used")
    processing_time_ms: int = Field(..., ge=0, description="Total processing time")
    failed_points: Optional[List[Dict[str, Any]]] = Field(None, description="Details of failed points")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional operation metadata")


class BatchUpsertRequest(BaseModel):
    """Batch upsert with flexible vector sizes per collection"""
    batches: List[Dict[str, Any]] = Field(..., description="List of batch operations")
    
    class BatchOperation(BaseModel):
        """Single batch operation"""
        collection_name: str = Field(..., description="Target collection")
        points: List[VectorPoint] = Field(..., description="Points to upsert")
        vector_size: Optional[int] = Field(None, description="Vector dimension")
        create_if_missing: bool = Field(False, description="Create collection if not exists")
    
    @validator('batches')
    def validate_batches(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one batch is required')
        return v


# =============================================
# HEALTH & MONITORING MODELS
# =============================================
