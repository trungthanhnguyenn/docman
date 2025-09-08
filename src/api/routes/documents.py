"""
Document management routes - CRUD operations for documents.
Supports the 4 core features:
1. Session management integration
2. Document CRUD operations  
3. Chunks management integration
4. Metrics and logging
"""
import uuid
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import StreamingResponse

from src.core.config import config
from src.core.models import (
    Document, DocumentMetadata
)
from src.api.services.database_manager import DatabaseManager
from src.core.exceptions import DatabaseConnectionException
from src.api.dependencies import get_database_manager


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/session/{session_id}/upload", response_model=Document)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> Document:
    """
    Upload document to a specific session (Core Feature: Document CRUD + Session integration).
    
    Args:
        session_id: Session identifier
        file: Uploaded file
        metadata: Optional document metadata as JSON string
        db_manager: Database manager instance
        
    Returns:
        Document: Uploaded document information
        
    Raises:
        HTTPException: If upload fails or session is invalid
    """
    try:
        # Generate document ID and user ID
        document_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())  # Random user ID as requested
        
        # Read file content
        file_content = await file.read()
        
        # Generate file hash
        file_hash = hashlib.md5(file_content).hexdigest()

        # content type
        if file.filename.endswith(".docx"):
            content_type = "docx"
        elif file.filename.endswith(".pdf"):
            content_type = "pdf"
        elif file.filename.endswith(".txt"):
            content_type = "txt"
        else:
            content_type = "unknown"

        # Create document metadata using the correct model structure
        doc_metadata = DocumentMetadata(
            document_id=document_id,
            filename=file.filename or f"document_{document_id}",
            file_size=len(file_content),
            content_type=content_type,
            file_hash=file_hash,
            chunks_count=0,
            processing_status="uploaded",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "custom_metadata": metadata
            }
        )
        
        # Upload document using database manager with correct method name
        result = await db_manager.create_document(
            file_data=file_content,
            filename=doc_metadata.filename,
            content_type=doc_metadata.content_type,
            file_hash=file_hash,
            document_id=document_id,
            metadata=doc_metadata.metadata
        )
        
        #! check for errors in result
        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload document: {result['error']}"
            )
        return Document(
            document_id=document_id,
            filename=doc_metadata.filename,
            content_type=doc_metadata.content_type,
            file_size=doc_metadata.file_size,
            upload_timestamp=doc_metadata.created_at or datetime.utcnow(),
            session_id=session_id,
            user_id=user_id,
            status="uploaded"
        )
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document info: {str(e)}"
        )

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> StreamingResponse:
    """
    FR006: Document Management - Download/retrieve original documents.
    
    Args:
        document_id: Document identifier
        db_manager: Database manager instance
        
    Returns:
        StreamingResponse: Document file stream
        
    Raises:
        HTTPException: If document not found or download fails
    """
    try:
        # Download document from MinIO
        result = await db_manager.download_document(document_id=document_id)
        
        if result.get("error"):
            if "not found" in result["error"].lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download document: {result['error']}"
                )
        
        # Create streaming response
        file_content = result["file_content"]
        filename = result["filename"]
        content_type = result["content_type"]
        
        # Create BytesIO stream
        from io import BytesIO
        file_stream = BytesIO(file_content)
        
        # Return streaming response with proper headers
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(file_content))
        }
        
        return StreamingResponse(
            file_stream,
            media_type=content_type,
            headers=headers
        )
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download document: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> dict:
    """
    Delete document from all databases (Core Feature: Document CRUD).
    
    Args:
        document_id: Document identifier
        db_manager: Database manager instance
        
    Returns:
        dict: Deletion status
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        result = await db_manager.delete_document(document_id=document_id)
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "deletion_results": result
        }
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.put("/{document_id}/metadata")
async def update_document_metadata(
    document_id: str,
    metadata: dict,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> dict:
    """
    FR006: Document Management - Update document metadata.
    
    Args:
        document_id: Document identifier
        metadata: New metadata
        db_manager: Database manager instance
        
    Returns:
        dict: Update status
        
    Raises:
        HTTPException: If update fails
    """
    try:
        result = await db_manager.update_document(
            document_id=document_id,
            updates=metadata
        )
        
        return {
            "message": "Document metadata updated successfully",
            "document_id": document_id,
            "update_result": result
        }
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update document metadata: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    document_id: Optional[str] = Query(None, description="Filter by document id"),
    filename_pattern: Optional[str] = Query(None, description="Filter by filename pattern"),
    include_metadata: bool = Query(True, description="Include detailed metadata"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> Dict[str, Any]:
    """
    FR006: Document Management - List all documents with filtering options.
    
    Args:
        filename_pattern: Optional pattern to filter filenames
        include_metadata: Whether to include detailed metadata
        limit: Maximum number of documents to return
        offset: Number of documents to skip (for pagination)
        db_manager: Database manager instance
        
    Returns:
        Dict: Documents list and pagination info
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        if not db_manager.minio_client:
            raise HTTPException(
                status_code=503,
                detail="MinIO client not available"
            )
        
        # Use MinIO search functionality
        search_params = {
            "bucket_name": config.minio.default_bucket,
            "include_metadata": include_metadata,
            "max_results": int(limit) + int(offset)  # Get more to handle offset
        }
        
        if filename_pattern:
            search_params["filename_pattern"] = filename_pattern

        if document_id:
            search_params["document_id"] = document_id
        
        result = db_manager.minio_client.search(**search_params)
        
        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list documents: {result['error']}"
            )
        
        documents = result.get("documents", [])
        
        # Apply offset and limit
        total_found = len(documents)
        paginated_documents = documents[int(offset):int(offset) + int(limit)]
        
        return {
            "documents": paginated_documents,
            "total_found": total_found,
            "returned_count": len(paginated_documents),
            "offset": offset,
            "limit": limit,
            "processing_time_ms": result.get("processing_time_ms", 0)
        }
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/check-duplicate/{file_hash}")
async def check_duplicate_document(
    file_hash: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> Dict[str, Any]:
    """
    FR002: Document Upload and Processing - Check for duplicate documents by hash.
    
    Args:
        file_hash: Hash of the file to check for duplicates
        db_manager: Database manager instance
        
    Returns:
        Dict: Information about duplicate document if found
        
    Raises:
        HTTPException: If check fails
    """
    try:
        if not db_manager.minio_client:
            raise HTTPException(
                status_code=503,
                detail="MinIO client not available"
            )
        
        # Check for duplicate using MinIO functionality
        duplicate_info = db_manager.minio_client.check_duplicate(
            file_hash=file_hash,
            bucket_name=config.minio.default_bucket
        )
        
        if duplicate_info:
            return {
                "duplicate_found": True,
                "existing_document": duplicate_info,
                "message": "Document with this hash already exists"
            }
        else:
            return {
                "duplicate_found": False,
                "message": "No duplicate found"
            }
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check for duplicate: {str(e)}"
        )


@router.get("/{document_id}/info")
async def get_document_info(
    document_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> Dict[str, Any]:
    """
    FR006: Document Management - Get detailed document information including MinIO metadata.
    
    Args:
        document_id: Document identifier
        db_manager: Database manager instance
        
    Returns:
        Dict: Detailed document information
        
    Raises:
        HTTPException: If document not found
    """
    try:
        if not db_manager.minio_client:
            raise HTTPException(
                status_code=503,
                detail="MinIO client not available"
            )
        
        # Get document info from MinIO
        document_info = db_manager.minio_client.get_document_info(
            document_id=document_id,
            bucket_name=config.minio.default_bucket
        )
        
        if document_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        return document_info
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document info: {str(e)}"
        )
    

@router.get("/{document_id}/metadata", response_model=DocumentMetadata)
async def get_document_metadata(
    document_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> DocumentMetadata:
    """
    FR006: Document Management - Get document metadata.
    
    Args:
        document_id: Document identifier
        db_manager: Database manager instance
        
    Returns:
        DocumentMetadata: Document metadata
        
    Raises:
        HTTPException: If document not found
    """
    try:
        # Get metadata from PostgreSQL first (authoritative source)
        metadata = await db_manager.get_document(document_id=document_id)
        
        if metadata is None:
            # Try getting from MinIO as fallback
            if not db_manager.minio_client:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}"
                )
            
            minio_info = db_manager.minio_client.get_document_info(
                document_id=document_id,
                bucket_name=db_manager.minio_client._client._bucket_name if hasattr(db_manager.minio_client._client, '_bucket_name') else 'documents'
            )
            
            if minio_info is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}"
                )
            
            # Convert MinIO info to DocumentMetadata format
            metadata = {
                "document_id": document_id,
                "filename": minio_info.get("filename", "unknown"),
                "file_size": minio_info.get("file_size", 0),
                "content_type": minio_info.get("content_type", "application/octet-stream"),
                "file_hash": minio_info.get("file_hash", ""),
                "chunks_count": 0,
                "processing_status": minio_info.get("processing_status", "stored"),
                "created_at": datetime.fromisoformat(minio_info["upload_time"]) if minio_info.get("upload_time") else datetime.utcnow(),
                "updated_at": datetime.fromisoformat(minio_info["last_modified"]) if minio_info.get("last_modified") else datetime.utcnow(),
                "metadata": minio_info.get("metadata", {})
            }
        
        # Return DocumentMetadata object
        return DocumentMetadata(**metadata)
        
    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document metadata: {str(e)}"
        )


@router.post("/session/{session_id}/upload-folder", response_model=List[Document])
async def upload_folder_documents(
    session_id: str,
    files: List[UploadFile] = File(...),
    relative_paths: str = Form(...),  # Changed to str to receive JSON
    metadata: Optional[str] = Form(None),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> List[Document]:
    """
    Upload multiple documents from a folder to a specific session with preserved folder structure.

    Args:
        session_id: Session identifier
        files: List of uploaded files from the folder
        relative_paths: JSON string of relative paths list
        metadata: Optional document metadata as JSON string
        db_manager: Database manager instance

    Returns:
        List[Document]: List of uploaded document information

    Raises:
        HTTPException: If upload fails or session is invalid
    """
    # Parse relative_paths from JSON string
    import json
    try:
        parsed_relative_paths = json.loads(relative_paths)
        if not isinstance(parsed_relative_paths, list):
            raise ValueError("relative_paths must be a list")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid relative_paths format: {str(e)}"
        )
    
    if len(files) != len(parsed_relative_paths):
        raise HTTPException(
            status_code=400,
            detail=f"Number of files ({len(files)}) must match number of relative paths ({len(parsed_relative_paths)})"
        )

    try:
        uploaded_documents = []

        for file, relative_path in zip(files, parsed_relative_paths):
            # Generate document ID and user ID
            document_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())

            # Read file content
            file_content = await file.read()

            # Generate file hash
            file_hash = hashlib.md5(file_content).hexdigest()

            # Determine content type
            filename = file.filename or f"document_{document_id}"
            if filename.endswith(".docx"):
                content_type = "docx"
            elif filename.endswith(".pdf"):
                content_type = "pdf"
            elif filename.endswith(".txt"):
                content_type = "txt"
            else:
                content_type = "unknown"

            # Create document metadata
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                filename=file.filename or f"document_{document_id}",
                file_size=len(file_content),
                content_type=content_type,
                file_hash=file_hash,
                chunks_count=0,
                processing_status="uploaded",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={
                    "session_id": session_id,
                    "user_id": user_id,
                    "relative_path": relative_path,  # Preserve folder structure
                    "custom_metadata": metadata
                }
            )

            # Upload document using database manager
            result = await db_manager.create_document(
                file_data=file_content,
                filename=doc_metadata.filename,
                content_type=doc_metadata.content_type,
                file_hash=file_hash,
                document_id=document_id,
                metadata=doc_metadata.metadata
            )

            # Check for errors in result
            if result.get("status") != "success":
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload document {file.filename}: {result}"
                )

            # Create Document response object with correct fields
            document_response = Document(
                document_id=document_id,
                filename=doc_metadata.filename,
                content_type=doc_metadata.content_type,
                file_size=doc_metadata.file_size,
                upload_timestamp=datetime.utcnow(),
                session_id=session_id,
                user_id=user_id,
                status="uploaded"
            )

            uploaded_documents.append(document_response)

        return uploaded_documents

    except DatabaseConnectionException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload folder documents: {str(e)}"
        )
