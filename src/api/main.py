# src/api/main.py
"""
Document Management System API

This FastAPI application provides endpoints for:
- Session management (CRUD)
- Document management (CRUD)
- Chunks management (CRUD)
- Health monitoring and metrics
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from src.core import config, DocumentManagementException, metrics
from src.api.routes import (
    documents,
    health,
    sessions,
    chunks,
    rag,
    chunking_v2
)
from src.api.services.database_manager import DatabaseManager
from src.api.services.model_lifecycle import get_model_lifecycle_manager
from src.api.services.embedding_service import cleanup_embedding_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global database manager instance
db_manager: DatabaseManager = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown"""
    global db_manager
    
    # Startup
    logger.info("🚀 Starting Document Management System API")
    
    try:
        # Initialize database connections
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Store in app state for route access
        app.state.db_manager = db_manager
        
        logger.info("✅ Database connections initialized successfully")
        logger.info(f"🔧 Configuration: {config.environment} environment")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database connections: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Document Management System API")
    
    # Cleanup ML models first to prevent semaphore leaks
    logger.info("🧹 Cleaning up ML models...")
    try:
        cleanup_embedding_service()
        lifecycle_manager = get_model_lifecycle_manager()
        lifecycle_manager.cleanup_all()
        logger.info("✅ ML models cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ ML models cleanup error: {e}")
    
    # Cleanup databases
    if db_manager:
        await db_manager.cleanup()
        logger.info("✅ Database connections closed successfully")

# Create FastAPI application
app = FastAPI(
    title="Document Management System API",
    description="Backend API for document upload, processing and management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record API metrics
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    metrics.record_document_operation(
        operation=f"{method}_{endpoint}",
        database="api",
        status="success" if status_code < 400 else "error",
        duration=process_time
    )
    
    return response

# Global exception handler
@app.exception_handler(DocumentManagementException)
async def document_management_exception_handler(request: Request, exc: DocumentManagementException):
    """Handle document management exceptions"""
    logger.error(f"Document management error: {exc.message}", extra=exc.details)
    
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__} if config.debug else {}
        }
    )

# Include routers - focusing on core features only
app.include_router(health.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chunks.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(chunking_v2.router, prefix="/api/v1")  # New chunking v2 routes

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Document Management System API",
        "version": "1.0.0",
        "status": "running",
        "environment": config.environment,
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

# API info endpoint
@app.get("/api/v1/info")
async def api_info():
    """Get API information and configuration"""
    return {
        "api": {
            "name": "Document Management System",
            "version": "1.0.0",
            "environment": config.environment
        },
        "features": {
            "session_management": True,
            "document_management": True,
            "chunks_management": True,
            "health_monitoring": True,
            "metrics_collection": True,
            "rag_pipeline": True,
            "server_side_embedding": True,
            "reranking": True
        },
        "limits": {
            "max_file_size_mb": config.minio.max_file_size // (1024 * 1024),
            "max_files_per_request": config.max_documents_per_request,
            "allowed_file_types": config.minio.allowed_extensions
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
