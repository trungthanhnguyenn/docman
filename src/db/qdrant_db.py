import os
import logging
import json
import uuid
import zipfile
from datetime import datetime
from typing import Optional, List, Any, Dict, Union

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from qdrant_client.http import models
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    models = None
    QdrantClient = None

from .interface import InterfaceDatabase

def get_distance_mapping():
    """Get distance mapping, only if Qdrant is available"""
    if not QDRANT_AVAILABLE or models is None:
        return {}
    return {
        'euclidean': models.Distance.EUCLID,
        'dot': models.Distance.DOT, 
        'manhattan': models.Distance.MANHATTAN,
        'cosine': models.Distance.COSINE
    }


class QdrantChunksDB(InterfaceDatabase):
    """ 
    Vector database using Qdrant for storing and searching document chunks.
    Supports the payload structure:
    {
        "document_id": "uuid",
        "doc_title": "AI Base in VietNam", 
        "page": 5,
        "chunk_content": "string",
        "file_url": "https://minio/bucket/path/file.pdf",
        "user_id": "None",
        "session_id": "None"
    }
    """
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client is not installed. Please install it with: pip install qdrant-client")
        self._client = self.connect_client(url, api_key=api_key)

    def connect_client(self, url, **kwargs) -> Any:
        """Connect to Qdrant client"""
        if not QDRANT_AVAILABLE or QdrantClient is None:
            return None
            
        api_key = kwargs.get('api_key')
        
        if url is not None and api_key is not None:
            try:
                # Cloud instance
                client = QdrantClient(url=url, api_key=api_key)
                # Test connection
                client.get_collections()
                logging.info(f"Successfully connected to Qdrant cloud at {url}")
                return client
            except Exception as e:
                logging.error(f"Failed to connect to Qdrant cloud: {e}")
                return None
        elif url is not None:
            try:
                # Local instance
                client = QdrantClient(url=url)
                # Test connection
                client.get_collections()
                logging.info(f"Successfully connected to Qdrant local at {url}")
                return client
            except Exception as e:
                logging.error(f"Failed to connect to Qdrant local: {e}")
                return None
        else:
            logging.error("Missing required connection parameters for Qdrant")
            return None
    
    def _check_client(self) -> bool:
        """Check if client is available"""
        if self._client is None:
            logging.error("Qdrant client is not connected")
            return False
        return True

    def create_collection(self, collection_name: str = "document_chunks", dimension: int = 768, distance: str = 'cosine') -> bool:
        """Create a collection if it doesn't exist"""
        if not self._check_client() or not QDRANT_AVAILABLE or models is None:
            return False
            
        try:
            distance_mapping = get_distance_mapping()
            distance_metric = distance_mapping.get(distance, distance_mapping.get('cosine'))
            
            if distance_metric is None:
                logging.error("Invalid distance metric and Qdrant models not available")
                return False
                
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, 
                    distance=distance_metric
                ),
            )
            logging.info(f"Collection '{collection_name}' created successfully")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logging.info(f"Collection '{collection_name}' already exists")
                return True
            logging.error(f"Error creating collection '{collection_name}': {e}")
            return False

    def insert(self, points: List[Dict[str, Any]], **kwargs) -> dict:
        """
        Insert document chunks into Qdrant collection.
        
        Args:
            points: List of dictionaries containing:
                - vectors: List[float] - embedding vector
                - payload: Dict containing document_id, doc_title, page, chunk_content, file_url, user_id, session_id
                - id: Optional[str] - point ID (will be generated if not provided)
        
        Returns:
            dict: Response with insertion results
        """
        import time
        start_time = time.time()
        
        collection_name = kwargs.get('collection_name', 'document_chunks')
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'points_processed': 0,
                'processing_time_ms': 0
            }
        
        # Ensure collection exists
        try:
            is_existed = self._client.collection_exists(collection_name)
            logging.info(f"Collection '{collection_name}' exists: {is_existed}")
        except Exception as e:
            dimension = kwargs.get('dimension', 768)
            distance = kwargs.get('distance', 'cosine')
            if not self.create_collection(collection_name, dimension, distance):
                return {
                    'status': 'failed',
                    'message': f'Failed to create collection {collection_name}',
                    'points_processed': 0,
                    'processing_time_ms': 0
                }

        # Prepare points for insertion
        qdrant_points = []
        failed_points = []
        
        for i, point in enumerate(points):
            try:
                # Extract required fields
                vector = point.get('vector')
                payload = point.get('payload', {})
                point_id = point.get('id') or str(uuid.uuid4())
                
                if not vector:
                    failed_points.append({
                        'index': i,
                        'error': 'Missing vector field'
                    })
                    continue
                
                # Validate required payload fields
                required_fields = ['document_id', 'chunk_content']
                missing_fields = [field for field in required_fields if not payload.get(field)]
                if missing_fields:
                    failed_points.append({
                        'index': i,
                        'error': f'Missing required payload fields: {missing_fields}'
                    })
                    continue
                
                # Ensure all payload fields are present with defaults
                validated_payload = {
                    'document_id': payload.get('document_id'),
                    'doc_title': payload.get('doc_title', ''),
                    'page': payload.get('page', 0),
                    'chunk_content': payload.get('chunk_content'),
                    'file_url': payload.get('file_url', ''),
                    'user_id': payload.get('user_id'),
                    'session_id': payload.get('session_id'),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                if models is None:
                    failed_points.append({
                        'index': i,
                        'error': 'Qdrant models not available'
                    })
                    continue
                
                qdrant_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=validated_payload
                ))
                
            except Exception as e:
                failed_points.append({
                    'index': i,
                    'error': f'Error processing point: {str(e)}'
                })
        
        # Insert points
        successful_count = 0
        try:
            if qdrant_points:
                self._client.upsert(
                    collection_name=collection_name,
                    points=qdrant_points
                )
                successful_count = len(qdrant_points)
                logging.info(f"Successfully inserted {successful_count} points into {collection_name}")
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error inserting points: {str(e)}',
                'points_processed': 0,
                'failed_points': failed_points,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = {
            'status': 'success',
            'points_processed': successful_count,
            'processing_time_ms': processing_time
        }
        
        if failed_points:
            response['failed_points'] = failed_points
        
        return response

    def search(self, **kwargs) -> dict:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            **kwargs:
                - query_vector: List[float] - query embedding vector
                - collection_name: str - collection to search in
                - limit: int - number of results to return
                - document_id: str - filter by specific document
                - user_id: str - filter by user
                - session_id: str - filter by session
                - page: int - filter by page number
        
        Returns:
            dict: Search results with chunks
        """
        import time
        start_time = time.time()
        
        query_vector = kwargs.get('query_vector')
        collection_name = kwargs.get('collection_name', 'document_chunks')
        limit = kwargs.get('limit', 5)
        
        if not query_vector:
            return {
                'status': 'failed',
                'message': 'Missing query_vector parameter',
                'chunks': [],
                'processing_time_ms': 0
            }
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'chunks': [],
                'processing_time_ms': 0
            }
        
        # Build filters
        filter_conditions = []
        
        if not QDRANT_AVAILABLE or models is None:
            return {
                'status': 'failed',
                'message': 'Qdrant models not available',
                'chunks': [],
                'processing_time_ms': 0
            }
        
        if kwargs.get('document_id'):
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=kwargs['document_id'])
                )
            )
        
        if kwargs.get('user_id'):
            filter_conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=kwargs['user_id'])
                )
            )
        
        if kwargs.get('session_id'):
            filter_conditions.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=kwargs['session_id'])
                )
            )
        
        if kwargs.get('page') is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="page",
                    match=models.MatchValue(value=kwargs['page'])
                )
            )
        
        # Perform search
        try:
            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
            )
            
            # Format results
            chunks = []
            for result in results:
                chunks.append({
                    "id": str(result.id),
                    "score": result.score,
                    "payload": result.payload
                })
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'status': 'success',
                'chunks': chunks,
                'total_found': len(chunks),
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Search error: {str(e)}',
                'chunks': [],
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def update(self, points: List[Dict[str, Any]], **kwargs) -> dict:
        """
        Update existing points in the collection.
        
        Args:
            points: List of dictionaries containing id, vector, and payload updates
            
        Returns:
            dict: Update results
        """
        import time
        start_time = time.time()
        
        collection_name = kwargs.get('collection_name', 'document_chunks')
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'points_updated': 0,
                'processing_time_ms': 0
            }
        
        updated_points = []
        failed_updates = []
        
        for i, point in enumerate(points):
            try:
                point_id = point.get('id')
                if not point_id:
                    failed_updates.append({
                        'index': i,
                        'error': 'Missing point ID'
                    })
                    continue
                
                # Prepare update data
                vector = point.get('vector')
                payload = point.get('payload', {})
                
                # Add update timestamp
                if payload:
                    payload['updated_at'] = datetime.utcnow().isoformat()
                
                if not QDRANT_AVAILABLE or models is None:
                    failed_updates.append({
                        'index': i,
                        'error': 'Qdrant models not available'
                    })
                    continue
                
                # Only create point if vector is provided
                if vector is not None:
                    qdrant_point = models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                    updated_points.append(qdrant_point)
                else:
                    # Skip updates without vectors for now
                    failed_updates.append({
                        'index': i,
                        'error': 'Vector required for update operation'
                    })
                    continue
                
            except Exception as e:
                failed_updates.append({
                    'index': i,
                    'error': f'Error processing update: {str(e)}'
                })
        
        # Perform updates
        successful_count = 0
        try:
            if updated_points:
                self._client.upsert(
                    collection_name=collection_name,
                    points=updated_points
                )
                successful_count = len(updated_points)
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error updating points: {str(e)}',
                'points_updated': 0,
                'failed_updates': failed_updates,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = {
            'status': 'success',
            'points_updated': successful_count,
            'processing_time_ms': processing_time
        }
        
        if failed_updates:
            response['failed_updates'] = failed_updates
        
        return response

    def delete(self, points_ids: Union[str, List[str]], **kwargs) -> dict:
        """
        Delete points by IDs or by document_id filter.
        
        Args:
            points_ids: Point IDs to delete, or document_id for filtering
            **kwargs:
                - collection_name: str - collection name
                - by_document_id: bool - if True, treat points_ids as document_ids to filter by
        
        Returns:
            dict: Deletion results
        """
        import time
        start_time = time.time()
        
        collection_name = kwargs.get('collection_name', 'document_chunks')
        by_document_id = kwargs.get('by_document_id', False)
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'processing_time_ms': 0
            }
        
        if isinstance(points_ids, str):
            points_ids = [points_ids]
        
        try:
            if by_document_id:
                if not QDRANT_AVAILABLE or models is None:
                    return {
                        'status': 'failed',
                        'message': 'Qdrant models not available',
                        'processing_time_ms': int((time.time() - start_time) * 1000)
                    }
                
                # Delete by document_id filter
                deleted_count = 0
                for doc_id in points_ids:
                    self._client.delete(
                        collection_name=collection_name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="document_id",
                                        match=models.MatchValue(value=doc_id)
                                    ),
                                ]
                            )
                        ),
                    )
                    deleted_count += 1
                
                return {
                    'status': 'success',
                    'message': f'Deleted chunks for {deleted_count} document(s)',
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            else:
                if not QDRANT_AVAILABLE or models is None:
                    return {
                        'status': 'failed',
                        'message': 'Qdrant models not available',
                        'processing_time_ms': int((time.time() - start_time) * 1000)
                    }
                
                # Delete by point IDs - cast to proper type
                from typing import cast, Any
                self._client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=cast(Any, points_ids)),
                )
                
                return {
                    'status': 'success',
                    'message': f'Deleted {len(points_ids)} point(s)',
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error deleting points: {str(e)}',
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def get_chunks_by_document_id(self, document_id: str, collection_name: str = 'document_chunks') -> dict:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID to filter by
            collection_name: Collection to search in
            
        Returns:
            dict: Document chunks
        """
        import time
        start_time = time.time()
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'chunks': [],
                'processing_time_ms': 0
            }
        
        try:
            if not QDRANT_AVAILABLE or models is None:
                return {
                    'status': 'failed',
                    'message': 'Qdrant models not available',
                    'chunks': [],
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            
            results = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", 
                            match=models.MatchValue(value=document_id)
                        ),
                    ],
                ),
            )
            
            # Format results
            chunks = []
            if results[0]:  # Check if results exist
                for chunk in results[0]:
                    chunks.append({
                        "id": str(chunk.id),
                        "payload": chunk.payload,
                    })
            
            return {
                'status': 'success',
                'chunks': chunks,
                'total_found': len(chunks),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error retrieving chunks: {str(e)}',
                'chunks': [],
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def get_all_chunks(self, collection_name: str = 'document_chunks', limit: int = 100) -> dict:
        """
        Get all chunks from collection with pagination.
        
        Args:
            collection_name: Collection to retrieve from
            limit: Maximum number of chunks to retrieve
            
        Returns:
            dict: All chunks with metadata
        """
        import time
        start_time = time.time()
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'chunks': [],
                'processing_time_ms': 0
            }
        
        try:
            results = self._client.scroll(
                collection_name=collection_name,
                limit=limit,
            )
            
            # Format results
            chunks = []
            if results[0]:  # Check if results exist
                for chunk in results[0]:
                    chunks.append({
                        "id": str(chunk.id),
                        "payload": chunk.payload,
                    })
            
            return {
                'status': 'success',
                'chunks': chunks,
                'total_found': len(chunks),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error retrieving chunks: {str(e)}',
                'chunks': [],
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def delete_collection(self, collection_name: str) -> dict:
        """
        Delete an entire collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            dict: Deletion result
        """
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected'
            }
        
        try:
            self._client.delete_collection(collection_name=collection_name)
            return {
                'status': 'success',
                'message': f"Collection '{collection_name}' deleted successfully"
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': f"Error deleting collection '{collection_name}': {str(e)}"
            }

    def get_collection_info(self, collection_name: str = 'document_chunks') -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of collection to inspect
            
        Returns:
            dict: Collection information
        """
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected'
            }
        
        try:
            info = self._client.get_collection(collection_name)
            return {
                'status': 'success',
                'collection_info': {
                    'points_count': info.points_count,
                    'status': info.status,
                    'vectors_count': info.vectors_count,
                    'config': {
                        'params': info.config.params.dict() if info.config.params else {},
                        'hnsw_config': info.config.hnsw_config.dict() if info.config.hnsw_config else {},
                        'optimizer_config': info.config.optimizer_config.dict() if info.config.optimizer_config else {}
                    }
                }
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': f"Error getting collection info: {str(e)}"
            }

    def list_collections(self) -> dict:
        """
        List all available collections.
        
        Returns:
            dict: List of collections
        """
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'collections': []
            }
        
        try:
            collections = self._client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            return {
                'status': 'success',
                'collections': collection_names,
                'total_collections': len(collection_names)
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': f"Error listing collections: {str(e)}",
                'collections': []
            }

    def backup_collection(self, collection_name: str, output_dir: str = "backups") -> dict:
        """
        Create a snapshot backup of a collection using Qdrant's native snapshot API.
        
        Args:
            collection_name: Name of collection to backup
            output_dir: Directory to save backup files
            
        Returns:
            dict: Backup result with snapshot info
        """
        import time
        start_time = time.time()
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'processing_time_ms': 0
            }
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create snapshot using Qdrant's API
            snapshot_result = self._client.create_snapshot(collection_name=collection_name)
            
            if not snapshot_result:
                return {
                    'status': 'failed',
                    'message': f'Failed to create snapshot for collection {collection_name}',
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            
            # Download the snapshot
            snapshot_url = f"{self._client._client.rest_uri}/collections/{collection_name}/snapshots/{snapshot_result.name}"
            
            # Use requests to download if available, otherwise try client download
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{collection_name}_snapshot_{timestamp}.tar"
            backup_filepath = os.path.join(output_dir, backup_filename)
            
            if REQUESTS_AVAILABLE and requests:
                try:
                    # Download using requests
                    headers = {}
                    if hasattr(self._client, '_client') and hasattr(self._client._client, 'api_key'):
                        api_key = self._client._client.api_key
                        if api_key:
                            headers['api-key'] = api_key
                    
                    response = requests.get(snapshot_url, headers=headers, stream=True)
                    response.raise_for_status()
                    
                    with open(backup_filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logging.info(f"Snapshot downloaded successfully to {backup_filepath}")
                    
                except Exception as download_error:
                    logging.warning(f"Failed to download snapshot via requests: {download_error}")
                    # Fallback: keep snapshot on server
                    backup_filepath = f"server://{collection_name}/snapshots/{snapshot_result.name}"
            else:
                # No requests available, keep snapshot on server
                backup_filepath = f"server://{collection_name}/snapshots/{snapshot_result.name}"
                logging.info(f"Snapshot created on server: {snapshot_result.name}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'status': 'success',
                'message': f'Snapshot backup completed for collection {collection_name}',
                'backup_file': backup_filepath,
                'snapshot_name': snapshot_result.name,
                'snapshot_size': getattr(snapshot_result, 'size', 'unknown'),
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error creating snapshot for collection {collection_name}: {str(e)}',
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def backup_multiple_collections(self, collection_names: List[str], output_dir: str = "backups") -> dict:
        """
        Create snapshots for multiple collections.
        
        Args:
            collection_names: List of collection names to backup
            output_dir: Directory to save backup files
            
        Returns:
            dict: Backup results for all collections
        """
        import time
        start_time = time.time()
        
        backup_results = []
        successful_backups = []
        failed_backups = []
        
        for collection_name in collection_names:
            result = self.backup_collection(collection_name, output_dir)
            backup_results.append({
                'collection_name': collection_name,
                'result': result
            })
            
            if result['status'] == 'success':
                successful_backups.append({
                    'collection_name': collection_name,
                    'backup_file': result['backup_file'],
                    'snapshot_name': result.get('snapshot_name')
                })
            else:
                failed_backups.append({
                    'collection_name': collection_name,
                    'error': result['message']
                })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'status': 'success' if not failed_backups else 'partial',
            'backup_results': backup_results,
            'successful_backups': successful_backups,
            'failed_backups': failed_backups,
            'total_collections': len(collection_names),
            'successful_count': len(successful_backups),
            'failed_count': len(failed_backups),
            'processing_time_ms': processing_time
        }

    def restore_collection(self, backup_path: str, new_collection_name: Optional[str] = None) -> dict:
        """
        Restore a collection from Qdrant snapshot backup.
        
        Args:
            backup_path: Path to snapshot file (.tar) or server snapshot reference
            new_collection_name: Optional new name for restored collection
            
        Returns:
            dict: Restore result
        """
        import time
        start_time = time.time()
        
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'processing_time_ms': 0
            }
        
        try:
            # Check if it's a server snapshot reference or local file
            if backup_path.startswith("server://"):
                # Server snapshot format: server://collection_name/snapshots/snapshot_name
                parts = backup_path.replace("server://", "").split("/")
                if len(parts) >= 3 and parts[1] == "snapshots":
                    original_collection = parts[0]
                    snapshot_name = parts[2]
                    
                    # Use server-side snapshot
                    collection_name = new_collection_name or original_collection
                    
                    # Restore from server snapshot
                    self._client.recover_snapshot(
                        collection_name=collection_name,
                        snapshot_path=f"/snapshots/{snapshot_name}"
                    )
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    return {
                        'status': 'success',
                        'message': f'Restored collection {collection_name} from server snapshot',
                        'collection_name': collection_name,
                        'snapshot_name': snapshot_name,
                        'processing_time_ms': processing_time
                    }
                else:
                    return {
                        'status': 'failed',
                        'message': 'Invalid server snapshot reference format',
                        'processing_time_ms': int((time.time() - start_time) * 1000)
                    }
            else:
                # Local snapshot file
                if not os.path.exists(backup_path):
                    return {
                        'status': 'failed',
                        'message': f'Backup file not found: {backup_path}',
                        'processing_time_ms': int((time.time() - start_time) * 1000)
                    }
                
                # Extract collection name from filename if not provided
                if new_collection_name is None:
                    filename = os.path.basename(backup_path)
                    # Expected format: collection_name_snapshot_timestamp.tar
                    if "_snapshot_" in filename:
                        new_collection_name = filename.split("_snapshot_")[0]
                    else:
                        return {
                            'status': 'failed',
                            'message': 'Cannot determine collection name from filename. Please provide new_collection_name.',
                            'processing_time_ms': int((time.time() - start_time) * 1000)
                        }
                
                # Upload and restore from local snapshot
                try:
                    # For local files, we need to upload the snapshot first
                    # This is a more complex operation that may require REST API calls
                    
                    # Check if collection already exists
                    existing_collections = self.list_collections()
                    if (existing_collections['status'] == 'success' and 
                        new_collection_name in existing_collections['collections']):
                        
                        # Delete existing collection
                        delete_result = self.delete_collection(new_collection_name)
                        if delete_result['status'] != 'success':
                            logging.warning(f"Failed to delete existing collection: {delete_result['message']}")
                    
                    # Use REST API to upload and restore snapshot
                    if REQUESTS_AVAILABLE and requests:
                        base_url = self._client._client.rest_uri
                        
                        # Prepare headers
                        headers = {}
                        if hasattr(self._client, '_client') and hasattr(self._client._client, 'api_key'):
                            api_key = self._client._client.api_key
                            if api_key:
                                headers['api-key'] = api_key
                        
                        # Upload snapshot
                        upload_url = f"{base_url}/collections/{new_collection_name}/snapshots/upload"
                        
                        with open(backup_path, 'rb') as f:
                            files = {'snapshot': f}
                            response = requests.post(upload_url, headers=headers, files=files)
                            response.raise_for_status()
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        return {
                            'status': 'success',
                            'message': f'Restored collection {new_collection_name} from snapshot',
                            'collection_name': new_collection_name,
                            'backup_file': backup_path,
                            'processing_time_ms': processing_time
                        }
                    else:
                        return {
                            'status': 'failed',
                            'message': 'requests library not available for snapshot upload',
                            'processing_time_ms': int((time.time() - start_time) * 1000)
                        }
                        
                except Exception as upload_error:
                    return {
                        'status': 'failed',
                        'message': f'Error uploading/restoring snapshot: {str(upload_error)}',
                        'processing_time_ms': int((time.time() - start_time) * 1000)
                    }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error restoring collection: {str(e)}',
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def list_snapshots(self, collection_name: str) -> dict:
        """
        List all snapshots for a collection.
        
        Args:
            collection_name: Name of collection to list snapshots for
            
        Returns:
            dict: List of snapshots
        """
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected',
                'snapshots': []
            }
        
        try:
            snapshots = self._client.list_snapshots(collection_name=collection_name)
            
            snapshot_list = []
            for snapshot in snapshots:
                snapshot_list.append({
                    'name': snapshot.name,
                    'size': getattr(snapshot, 'size', 'unknown'),
                    'creation_time': getattr(snapshot, 'creation_time', 'unknown')
                })
            
            return {
                'status': 'success',
                'collection_name': collection_name,
                'snapshots': snapshot_list,
                'total_snapshots': len(snapshot_list)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error listing snapshots for {collection_name}: {str(e)}',
                'snapshots': []
            }

    def delete_snapshot(self, collection_name: str, snapshot_name: str) -> dict:
        """
        Delete a specific snapshot.
        
        Args:
            collection_name: Name of collection
            snapshot_name: Name of snapshot to delete
            
        Returns:
            dict: Deletion result
        """
        if not self._check_client():
            return {
                'status': 'failed',
                'message': 'Qdrant client not connected'
            }
        
        try:
            self._client.delete_snapshot(
                collection_name=collection_name,
                snapshot_name=snapshot_name
            )
            
            return {
                'status': 'success',
                'message': f'Snapshot {snapshot_name} deleted from collection {collection_name}'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error deleting snapshot {snapshot_name}: {str(e)}'
            }
