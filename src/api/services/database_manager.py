# src/api/services/database_manager.py
"""
Database Manager Service

Centralized management of database connections and operations.
Supports:
- Session management (CRUD)
- Document management (CRUD) 
- Chunks management (CRUD)
- Metrics and logging
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.core import (
    config,
    DatabaseConnectionException,
    metrics
)

from src.db.minio_db import MinioDB
from src.db.qdrant_db import QdrantChunksDB
from src.db.postgres_db import PostgresDB

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database manager supporting:
    - Session management (CRUD)
    - Document management (CRUD)
    - Chunks management (CRUD)
    - Metrics and logging
    """
    
    def __init__(self):
        self.minio_client: Optional[MinioDB] = None
        self.qdrant_client: Optional[QdrantChunksDB] = None
        self.postgres_client: Optional[PostgresDB] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all database connections with metrics and logging"""
        logger.info("🔌 Initializing database connections...")
        
        try:
            # Initialize MinIO for documents
            await self._init_minio()
            
            # Initialize Qdrant for chunks
            await self._init_qdrant()
            
            # Initialize PostgreSQL for sessions and metadata
            await self._init_postgres()
            
            self._initialized = True
            logger.info("✅ All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize databases: {e}")
            await self.cleanup()
            raise DatabaseConnectionException("all", {"error": str(e)})
    
    async def _init_minio(self):
        """Initialize MinIO connection for document storage"""
        try:
            logger.info("🗄️ Connecting to MinIO...")
            
            self.minio_client = MinioDB(
                endpoint=config.minio.endpoint,
                access_key=config.minio.access_key,
                secret_key=config.minio.secret_key,
                secure=config.minio.secure
            )
            
            # Test connection and create bucket
            if self.minio_client._client:
                success = self.minio_client.create_bucket(config.minio.default_bucket)
                if success:
                    logger.info("✅ MinIO connection established")
                    metrics.record_database_connection("minio", 1)
                else:
                    raise DatabaseConnectionException("MinIO", {"endpoint": config.minio.endpoint})
            else:
                raise DatabaseConnectionException("MinIO", {"endpoint": config.minio.endpoint})
                
        except Exception as e:
            logger.error(f"❌ MinIO connection failed: {e}")
            metrics.record_database_error("minio", "connection_failed")
            raise DatabaseConnectionException("MinIO", {"endpoint": config.minio.endpoint, "error": str(e)})
    
    async def _init_qdrant(self):
        """Initialize Qdrant connection for chunk storage"""
        try:
            logger.info("🔍 Connecting to Qdrant...")
            
            self.qdrant_client = QdrantChunksDB(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key
            )
            
            # Test connection and create collection
            if self.qdrant_client._client:
                success = self.qdrant_client.create_collection(
                    collection_name=config.qdrant.default_collection_name,
                    dimension=config.qdrant.vector_dimension,
                    distance=config.qdrant.distance_metric
                )
                if success:
                    logger.info("✅ Qdrant connection established")
                    metrics.record_database_connection("qdrant", 1)
                else:
                    raise DatabaseConnectionException("Qdrant", {"url": config.qdrant.url})
            else:
                raise DatabaseConnectionException("Qdrant", {"url": config.qdrant.url})
                
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            metrics.record_database_error("qdrant", "connection_failed")
            raise DatabaseConnectionException("Qdrant", {"url": config.qdrant.url, "error": str(e)})
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection for sessions and metadata"""
        try:
            logger.info("🐘 Connecting to PostgreSQL...")
            
            self.postgres_client = PostgresDB(
                host=config.postgres.host,
                port=config.postgres.port,
                database=config.postgres.database,
                user=config.postgres.user,
                password=config.postgres.password
            )
            
            # Test connection
            if self.postgres_client._connection:
                logger.info("✅ PostgreSQL connection established")
                metrics.record_database_connection("postgres", 1)
            else:
                raise DatabaseConnectionException("PostgreSQL", {"host": config.postgres.host, "database": config.postgres.database})
                
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            metrics.record_database_error("postgres", "connection_failed")
            raise DatabaseConnectionException("PostgreSQL", {"host": config.postgres.host, "database": config.postgres.database, "error": str(e)})
    
    async def cleanup(self):
        """Cleanup all database connections with metrics logging"""
        logger.info("🧹 Cleaning up database connections...")
        
        # Record final metrics
        metrics.record_database_connection("minio", 0)
        metrics.record_database_connection("qdrant", 0)
        metrics.record_database_connection("postgres", 0)
        
        self._initialized = False
        logger.info("✅ Database cleanup completed")
    
    def is_healthy(self) -> Dict[str, bool]:
        """Check health status of all databases"""
        health_status = {
            "minio": False,
            "qdrant": False,
            "postgres": False,
            "overall": False
        }
        
        try:
            # Check MinIO
            if self.minio_client and self.minio_client._check_client():
                health_status["minio"] = True
            
            # Check Qdrant
            if self.qdrant_client and self.qdrant_client._check_client():
                health_status["qdrant"] = True
            
            # Check PostgreSQL
            if self.postgres_client and self.postgres_client._check_connection():
                health_status["postgres"] = True
            
            # Overall health
            health_status["overall"] = all([
                health_status["minio"],
                health_status["qdrant"],
                health_status["postgres"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    # =============================================
    # DOCUMENT MANAGEMENT (CRUD)
    # =============================================
    
    async def create_document(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        file_hash: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create/upload document to MinIO and store metadata in PostgreSQL"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            # Store file in MinIO
            if not self.minio_client:
                raise DatabaseConnectionException("MinIO", {"reason": "client_not_initialized"})
            
            minio_points = [{
                'document_id': document_id,
                'file_data': file_data,
                'filename': filename,
                'file_size': len(file_data),
                'content_type': content_type,
                'file_hash': file_hash,
                'relative_path': metadata.get('relative_path') if metadata else None
            }]
            
            minio_result = self.minio_client.insert(
                points=minio_points,
                bucket_name=config.minio.default_bucket
            )
            
            if minio_result.get('documents'):
                # Store metadata in PostgreSQL
                if not self.postgres_client:
                    raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
                
                data = minio_result.get('documents', [])
                postgres_points = [{
                    'document_id': e['document_id'],
                    'user_id': metadata.get('user_id', 'anonymous') if metadata else 'anonymous',
                    'filename': e['filename'],
                    'file_size': e['file_size'],
                    'file_url': e['file_url'],
                    'file_type': content_type.split('/')[-1] if content_type else None,
                    'processing_status': e.get('processing_status', 'uploaded'),
                    'chunks_count': e.get('chunks_count', 0),
                    'created_at': e.get('created_at', datetime.utcnow()),
                    'updated_at': e.get('updated_at', datetime.utcnow()),
                    'metadata': metadata or {}
                } for e in data]
                
                postgres_result = self.postgres_client.insert(points=postgres_points)
                
                # Record metrics
                duration = (datetime.utcnow() - start_time).total_seconds()
                metrics.record_document_operation(
                    operation="create_document",
                    database="minio+postgres",
                    status="success",
                    duration=duration,
                    document_count=1
                )
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "minio_result": minio_result,
                    "postgres_result": postgres_result
                }
            else:
                raise Exception("MinIO upload failed")
                
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="create_document",
                database="minio+postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Document creation failed: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata from PostgreSQL"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.search(
                filters={"document_id": document_id},
                limit=1
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_document",
                database="postgres",
                status="success" if result.get("documents") else "not_found",
                duration=duration,
                document_count=1 if result.get("documents") else 0
            )
            
            if result.get("documents"):
                return result["documents"][0]
            return None
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_document",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Get document metadata failed: {e}")
            raise
    
    async def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update document metadata in PostgreSQL"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            points = [{
                "document_id": document_id,
                "updated_at": datetime.utcnow(),
                **updates
            }]
            
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            result = self.postgres_client.update(points=points)
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_document",
                database="postgres",
                status="success" if result.get("updated") else "error",
                duration=duration,
                document_count=1
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_document",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Document metadata update failed: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document from all databases (MinIO, Qdrant, PostgreSQL)"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        results = {}
        
        try:
            # Delete from MinIO
            if self.minio_client:
                minio_result = self.minio_client.delete([document_id])
                results["minio"] = minio_result
            
            # Delete chunks from Qdrant
            if self.qdrant_client:
                qdrant_result = self.qdrant_client.delete(
                    points_ids=[document_id],
                    by_document_id=True,
                    collection_name=config.qdrant.default_collection_name
                )
                results["qdrant"] = qdrant_result
            
            # Delete metadata from PostgreSQL
            if self.postgres_client:
                postgres_result = self.postgres_client.delete([document_id])
                results["postgres"] = postgres_result
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="delete_document",
                database="all",
                status="success",
                duration=duration,
                document_count=1
            )
            
            return {
                "status": "success",
                "document_id": document_id,
                "results": results
            }
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="delete_document",
                database="all",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Document deletion failed: {e}")
            results["error"] = str(e)
            return {
                "status": "failed",
                "results": results
            }
    
    async def download_document(self, document_id: str) -> Dict[str, Any]:
        """Download document content from MinIO"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.minio_client:
                raise DatabaseConnectionException("MinIO", {"reason": "client_not_initialized"})
            
            # Get document content from MinIO
            file_content = self.minio_client.get_file(
                document_id=document_id,
                bucket_name=config.minio.default_bucket
            )
            
            if file_content is None:
                # Record metrics for not found
                duration = (datetime.utcnow() - start_time).total_seconds()
                metrics.record_document_operation(
                    operation="download_document",
                    database="minio",
                    status="not_found",
                    duration=duration,
                    document_count=0
                )
                return {
                    "error": "Document not found",
                    "document_id": document_id
                }
            
            # Get document metadata from MinIO
            document_info = self.minio_client.get_document_info(
                document_id=document_id,
                bucket_name=config.minio.default_bucket
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="download_document",
                database="minio",
                status="success",
                duration=duration,
                document_count=1
            )
            
            return {
                "file_content": file_content,
                "filename": document_info.get("filename", "unknown") if document_info else "unknown",
                "content_type": document_info.get("content_type", "application/octet-stream") if document_info else "application/octet-stream",
                "file_size": document_info.get("file_size", len(file_content)) if document_info else len(file_content),
                "document_id": document_id
            }
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="download_document",
                database="minio",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Document download failed for {document_id}: {e}")
            raise DatabaseConnectionException("MinIO", {"operation": "download_document", "error": str(e)})

    # =============================================
    # CHUNKS MANAGEMENT (CRUD)
    # =============================================
    
    async def create_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create/store document chunks in Qdrant"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        collection = collection_name or config.qdrant.default_collection_name
        start_time = datetime.utcnow()
        
        try:
            if not self.qdrant_client:
                raise DatabaseConnectionException("Qdrant", {"reason": "client_not_initialized"})
            
            result = self.qdrant_client.insert(
                points=chunks,
                collection_name=collection
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="create_chunks",
                database="qdrant",
                status="success" if result.get("upserted") else "error",
                duration=duration,
                document_count=len(chunks)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="create_chunks",
                database="qdrant",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Chunk creation failed: {e}")
            raise
    
    async def get_chunks(
        self,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get/search chunks using vector similarity or filters"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        collection = collection_name or config.qdrant.default_collection_name
        start_time = datetime.utcnow()
        
        try:
            if not self.qdrant_client:
                raise DatabaseConnectionException("Qdrant", {"reason": "client_not_initialized"})
            
            if query_vector:
                # Vector similarity search
                search_params = {
                    "query_vector": query_vector,
                    "collection_name": collection,
                    "limit": limit
                }
                
                # Add filters if provided
                if filters:
                    search_params.update(filters)
                
                result = self.qdrant_client.search(**search_params)
            else:
                # Filter-based search - use search with empty vector or scroll
                result = self.qdrant_client.search(
                    query_vector=[0.0] * config.qdrant.vector_dimension,  # Dummy vector for filter-only search
                    collection_name=collection,
                    limit=limit,
                    filters=filters
                )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_chunks",
                database="qdrant",
                status="success",
                duration=duration,
                document_count=len(result.get("points", []))
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_chunks",
                database="qdrant",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Chunk retrieval failed: {e}")
            raise
    
    async def update_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update existing chunks in Qdrant"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        collection = collection_name or config.qdrant.default_collection_name
        start_time = datetime.utcnow()
        
        try:
            if not self.qdrant_client:
                raise DatabaseConnectionException("Qdrant", {"reason": "client_not_initialized"})
            
            # Qdrant handles updates through upsert
            result = self.qdrant_client.insert(
                points=chunks,
                collection_name=collection
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_chunks",
                database="qdrant",
                status="success" if result.get("upserted") else "error",
                duration=duration,
                document_count=len(chunks)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_chunks",
                database="qdrant",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Chunk update failed: {e}")
            raise
    
    async def delete_chunks(
        self,
        chunk_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete chunks by IDs or by document ID"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        collection = collection_name or config.qdrant.default_collection_name
        start_time = datetime.utcnow()
        
        try:
            if not self.qdrant_client:
                raise DatabaseConnectionException("Qdrant", {"reason": "client_not_initialized"})
            
            if document_id:
                # Delete all chunks for a document
                result = self.qdrant_client.delete(
                    points_ids=[document_id],
                    by_document_id=True,
                    collection_name=collection
                )
            elif chunk_ids:
                # Delete specific chunks
                result = self.qdrant_client.delete(
                    points_ids=chunk_ids,
                    by_document_id=False,
                    collection_name=collection
                )
            else:
                raise ValueError("Either chunk_ids or document_id must be provided")
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            deleted_count = len(chunk_ids) if chunk_ids else 1
            metrics.record_document_operation(
                operation="delete_chunks",
                database="qdrant",
                status="success" if result.get("deleted") else "error",
                duration=duration,
                document_count=deleted_count
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="delete_chunks",
                database="qdrant",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Chunk deletion failed: {e}")
            raise

    
    # =============================================
    # SESSION MANAGEMENT (CRUD)
    # =============================================
    
    async def create_session(
        self,
        user_id: str,
        expires_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        temp_collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session for chat history and document management"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.create_session(
                user_id=user_id,
                expires_at=expires_at,
                metadata=metadata,
                temp_collection_name=temp_collection_name
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="create_session",
                database="postgres",
                status="success" if result.get("session") else "error",
                duration=duration,
                document_count=1
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="create_session",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to create session: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "create_session", "error": str(e)})
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information by ID"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.get_session(session_id)
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_session",
                database="postgres",
                status="success" if result else "not_found",
                duration=duration,
                document_count=1 if result else 0
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_session",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to get session {session_id}: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "get_session", "error": str(e)})
    
    async def get_user_sessions(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get sessions for a specific user"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.get_user_sessions(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_user_sessions",
                database="postgres",
                status="success" if not result.get("error") else "error",
                duration=duration,
                document_count=result.get("returned_count", 0)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_user_sessions",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "get_user_sessions", "error": str(e)})
    
    async def update_session(
        self,
        session_id: str,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temp_collection_name: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Update session information"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.update_session(
                session_id=session_id,
                status=status,
                metadata=metadata,
                temp_collection_name=temp_collection_name,
                expires_at=expires_at
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_session",
                database="postgres",
                status="success" if result.get("session") else "error",
                duration=duration,
                document_count=1
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="update_session",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to update session {session_id}: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "update_session", "error": str(e)})
    
    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.delete_session(session_id)
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="delete_session",
                database="postgres",
                status="success" if result.get("deleted") else "error",
                duration=duration,
                document_count=1
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="delete_session",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "delete_session", "error": str(e)})
    
    async def expire_old_sessions(self) -> Dict[str, Any]:
        """Mark expired sessions as 'expired' based on expires_at timestamp"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.expire_old_sessions()
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="expire_sessions",
                database="postgres",
                status="success" if not result.get("error") else "error",
                duration=duration,
                document_count=result.get("expired_count", 0)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="expire_sessions",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to expire old sessions: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "expire_sessions", "error": str(e)})
    
    async def get_session_documents(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all documents for a specific session"""
        if not self._initialized:
            raise DatabaseConnectionException("system", {"reason": "not_initialized"})
        
        start_time = datetime.utcnow()
        
        try:
            if not self.postgres_client:
                raise DatabaseConnectionException("PostgreSQL", {"reason": "client_not_initialized"})
            
            result = self.postgres_client.get_session_documents(
                session_id=session_id,
                limit=limit,
                offset=offset
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_session_documents",
                database="postgres",
                status="success" if not result.get("error") else "error",
                duration=duration,
                document_count=result.get("returned_count", 0)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_document_operation(
                operation="get_session_documents",
                database="postgres",
                status="error",
                duration=duration,
                document_count=0
            )
            logger.error(f"Failed to get session documents for {session_id}: {e}")
            raise DatabaseConnectionException("PostgreSQL", {"operation": "get_session_documents", "error": str(e)})

    # =============================================
    # SYSTEM MONITORING & METRICS
    # =============================================
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with metrics logging"""
        stats = {
            "databases": self.is_healthy(),
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "session_management": True,
                "document_management": True,
                "chunks_management": True,
                "metrics_logging": True
            }
        }
        
        try:
            # Get collection info if available
            if self.qdrant_client and self.qdrant_client._check_client():
                collection_info = self.qdrant_client.get_collection_info(
                    config.qdrant.default_collection_name
                )
                stats["qdrant_collection"] = collection_info
        
        except Exception as e:
            logger.warning(f"Could not get system stats: {e}")
            stats["error"] = str(e)
        
        return stats