import aiohttp
import os
from typing import Optional, Dict, Any, List
import asyncio
import requests
import json
from datetime import datetime
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# For extracting text from docx files
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    print("Warning: python-docx not installed. Install with: pip install python-docx")
    DOCX_AVAILABLE = False

API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:8000")

def _create_session(user_id: str, expires_in_hours: Optional[int] = 24, metadata: Optional[Dict[str, Any]] = None, temp_collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Create session by user ID
    using docman POST /api/v1/sessions endpoint
    """
    session_data = {
        "user_id": user_id,
        "expires_in": expires_in_hours,
        "metadata": metadata,
        "temp_collection_name": temp_collection_name
    }
    response = requests.post(f"{API_BASE_URL}/api/v1/sessions", json=session_data)
    return response.json()

async def _upload_folder_batch(session_id: str, file_paths: List[str], relative_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Upload multiple documents as a batch to preserve folder structure
    using docman POST /documents/session/{session_id}/upload-folder endpoint
    """
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        
        # Open all files and keep them open
        opened_files = []
        for file_path in file_paths:
            file_obj = open(file_path, "rb")
            opened_files.append(file_obj)
            data.add_field('files', file_obj, filename=os.path.basename(file_path))

        # Add relative paths as JSON string
        data.add_field('relative_paths', json.dumps(relative_paths))

        # Add metadata if provided
        if metadata:
            data.add_field('metadata', str(metadata))
        
        try:
            async with session.post(f"{API_BASE_URL}/api/v1/documents/session/{session_id}/upload-folder", data=data) as response:
                response_text = await response.text()
                print(f"Batch {len(file_paths)} files - Response status: {response.status}")
                if response.status != 200:
                    print(f"HTTP Error {response.status}: {response_text}")
                return await response.json()
        except Exception as e:
            print(f"Request failed: {str(e)}")
            raise
        finally:
            # Ensure all files are closed
            for f in opened_files:
                f.close()

async def upload_folder_optimized(session_id: str, folder_path: str, metadata: Optional[Dict[str, Any]] = None, max_batch_size: int = 5, max_total_size_mb: float = 50.0) -> List[Dict[str, Any]]:
    """Upload folder with intelligent batching based on file sizes"""
    results = []
    
    # Collect files with their sizes
    file_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            file_info.append({
                'path': file_path,
                'relative_path': relative_path,
                'size_mb': file_size
            })
    
    # Sort by size (small files first for better packing)
    file_info.sort(key=lambda x: x['size_mb'])
    
    # Create intelligent batches
    batches = []
    current_batch = []
    current_size = 0.0
    
    for file in file_info:
        # Check if adding this file would exceed limits
        if len(current_batch) >= max_batch_size or (current_size + file['size_mb']) > max_total_size_mb:
            if current_batch:  # Save current batch if not empty
                batches.append(current_batch)
                current_batch = []
                current_size = 0.0
        
        current_batch.append(file)
        current_size += file['size_mb']
    
    # Add remaining batch
    if current_batch:
        batches.append(current_batch)
    
    print(f"Created {len(batches)} optimized batches")
    
    # Upload each batch
    for i, batch in enumerate(batches):
        batch_files = [f['path'] for f in batch]
        batch_relative_paths = [f['relative_path'] for f in batch]
        batch_size_mb = sum(f['size_mb'] for f in batch)
        
        print(f"Uploading batch {i+1}/{len(batches)}: {len(batch)} files, {batch_size_mb:.1f}MB")
        
        try:
            batch_result = await _upload_folder_batch(session_id, batch_files, batch_relative_paths, metadata)
            results.append({
                "batch": f"{i+1}",
                "files": batch_relative_paths,
                "total_size_mb": batch_size_mb,
                "result": batch_result
            })
        except Exception as e:
            print(f"Batch {i+1} failed: {str(e)}")
            print(f"   Files in batch: {batch_relative_paths}")
            results.append({
                "batch": f"{i+1}",
                "files": batch_relative_paths,
                "total_size_mb": batch_size_mb,
                "error": str(e)
            })
        
        # Small delay between batches to prevent overwhelming server
        await asyncio.sleep(1)  # Increased delay
    
    return results

def get_documents_by_session(session_id: str, limit: int = 500, off_set: int = 0) -> List[Dict[str, Any]]:
    """Get documents for a specific session"""
    response = requests.get(f"{API_BASE_URL}/api/v1/sessions/{session_id}/documents", params={
        "limit": limit,
        "off_set": off_set
    })
    if response.status_code == 200:
        return response.json().get("documents", [])
    else:
        print(f"Failed to get documents: {response.text}")
        return []

def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Get all sessions for a user"""
    response = requests.get(f"{API_BASE_URL}/api/v1/sessions/users/{user_id}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get sessions: {response.text}")
        return []

def save_session_info(user_id: str, session_id: str, description: str = None):
    """Save session info to a local file for easy retrieval"""
    session_file = f".docman_sessions_{user_id}.json"
    
    # Load existing sessions
    try:
        with open(session_file, 'r') as f:
            sessions = json.load(f)
    except:
        sessions = {}
    
    # Add/update session
    sessions[session_id] = {
        "user_id": user_id,
        "session_id": session_id,
        "description": description or f"Session created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat()
    }
    
    # Save back to file
    with open(session_file, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    print(f"Session info saved to {session_file}")

def load_saved_sessions(user_id: str) -> Dict[str, Any]:
    """Load saved session info from local file"""
    session_file = f".docman_sessions_{user_id}.json"
    
    try:
        with open(session_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def list_my_sessions(user_id: str, with_documents: bool = True):
    """List all available sessions for a user with document counts"""
    print(f"\n=== Sessions for user: {user_id} ===")
    
    # Load saved sessions
    saved_sessions = load_saved_sessions(user_id)
    if saved_sessions:
        print("\n--- Saved Sessions (Local) ---")
        for session_id, info in saved_sessions.items():
            print(f"Session: {session_id}")
            print(f"  Description: {info.get('description', 'No description')}")
            print(f"  Created: {info.get('created_at', 'Unknown')}")
            print()
    
    # Get sessions from server
    if with_documents:
        sessions_with_docs = find_session_with_documents(user_id)
        if sessions_with_docs:
            print("--- Active Sessions (With Documents) ---")
            for item in sessions_with_docs:
                session = item["session"]
                print(f"Session: {session['session_id']}")
                print(f"  Status: {session.get('status', 'unknown')}")
                print(f"  Created: {session.get('created_at', 'unknown')}")
                print(f"  Documents: {item['document_count']}")
                print(f"  Sample files: {[doc.get('filename') for doc in item['sample_documents']]}")
                print()
        else:
            print("No sessions with documents found")
    else:
        sessions = get_user_sessions(user_id)
        if sessions:
            print("--- All Sessions ---")
            for session in sessions:
                print(f"Session: {session['session_id']}")
                print(f"  Status: {session.get('status', 'unknown')}")
                print(f"  Created: {session.get('created_at', 'unknown')}")
                print()

def extract_text_from_docx(content_bytes: bytes) -> str:
    """Extract text content from .docx file bytes"""
    if not DOCX_AVAILABLE:
        return "Error: python-docx library not available. Install with: pip install python-docx"
    
    try:
        # Create a BytesIO object from bytes
        doc_stream = BytesIO(content_bytes)
        
        # Load the document
        doc = DocxDocument(doc_stream)
        
        # Extract text from all paragraphs
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                text_content.append(paragraph.text.strip())
        
        # Join paragraphs with newlines
        full_text = '\n'.join(text_content)
        
        # If no text found, check tables
        if not full_text.strip():
            table_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        table_text.append(' | '.join(row_text))
            
            if table_text:
                full_text = '\n'.join(table_text)
        
        return full_text if full_text.strip() else "No text content found in document"
        
    except Exception as e:
        return f"Error extracting text from docx: {str(e)}"

def get_documents_content_by_indicators(session_id: str, indicators: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Get document contents by category and index indicators
    
    Args:
        session_id: Session identifier
        indicators: List of dicts with "category" and "index" keys
                   e.g. [{"category": "ami", "index": "1"}, {"category": "bci_1", "index": "1"}]
    
    Returns:
        List of dicts with "category" and "content" keys
        e.g. [{"category": "ami", "content": "Actual text content from file..."}, {"category": "bci_1", "content": "Actual text content..."}]
    """
    # Step 1: Get all documents in the session
    documents = get_documents_by_session(session_id)
    
    if not documents:
        print(f"No documents found in session {session_id}")
        return []
    
    # Step 2: Create lookup map from documents
    doc_lookup = {}
    for doc in documents:
        relative_path = doc.get('metadata', {}).get('relative_path', '')
        document_id = doc.get('document_id')
        filename = doc.get('filename', '')
        
        if relative_path and document_id:
            # Extract category and index from relative_path
            path_parts = relative_path.split('/')
            if len(path_parts) >= 2:
                folder_name = path_parts[-2]  # "ami", "bci_1", etc.
                filename_without_ext = path_parts[-1].split('.')[0]  # "ami_1", "bci_1", etc.
                
                if '_' in filename_without_ext:
                    file_prefix, file_index = filename_without_ext.split('_', 1)
                    
                    # Fixed logic for both patterns
                    # Pattern 1: ami/ami_1.docx -> key: ami_1 (category=ami, index=1) 
                    # Pattern 2: bci_3/bci_3.docx -> key: bci_3_3 (category=bci_3, index=3)
                    
                    if folder_name == file_prefix:
                        # Simple pattern: folder matches file prefix
                        key = f"{folder_name}_{file_index}"  # ami_1
                        category = folder_name  # ami
                        index = file_index      # 1
                    else:
                        # Numbered folder pattern: folder has number, file might have different number
                        key = f"{folder_name}_{file_index}"  # bci_3_3 
                        category = folder_name  # bci_3
                        index = file_index      # 3
                    
                    doc_lookup[key] = {
                        'document_id': document_id,
                        'category': category,
                        'index': index,
                        'filename': filename,
                        'relative_path': relative_path
                    }
    
    print(f"Created lookup map with {len(doc_lookup)} entries")
    
    # Step 3: Find and download content for each indicator
    results = []
    
    for indicator in indicators:
        category = indicator.get('category')
        index = indicator.get('index')
        
        if not category or not index:
            results.append({
                'category': category or 'unknown',
                'content': f'Error: Invalid indicator {indicator}'
            })
            continue
        
        # Create lookup key
        lookup_key = f"{category}_{index}"
        
        if lookup_key in doc_lookup:
            doc_info = doc_lookup[lookup_key]
            document_id = doc_info['document_id']
            
            try:
                # Download document content
                print(f"Downloading {category}_{index} (ID: {document_id})...")
                response = requests.get(f"{API_BASE_URL}/api/v1/documents/{document_id}/download")
                
                if response.status_code == 200:
                    content_bytes = response.content
                    content_size = len(content_bytes)
                    
                    # Extract text based on file type
                    if doc_info['filename'].endswith('.txt'):
                        try:
                            content = content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                content = content_bytes.decode('latin-1')
                            except:
                                content = f"Error: Cannot decode text file ({content_size} bytes)"
                    
                    elif doc_info['filename'].endswith('.docx'):
                        # Extract text from docx
                        content = extract_text_from_docx(content_bytes)
                    
                    elif doc_info['filename'].endswith('.pdf'):
                        # For PDF files, you could use PyPDF2 or similar
                        content = f"PDF file detected ({content_size} bytes). Text extraction from PDF not implemented. Install PyPDF2 for PDF support."
                    
                    else:
                        # For other file types
                        content = f"Unsupported file type ({doc_info['filename']}, {content_size} bytes). Cannot extract text."
                    
                    results.append({
                        'category': category,
                        'content': content,
                        'document_id': document_id,
                        'filename': doc_info['filename'],
                        'relative_path': doc_info['relative_path'],
                        'file_size': content_size
                    })
                    
                    print(f"✓ Downloaded and extracted text from {category}_{index}: {len(content)} characters")
                    
                else:
                    results.append({
                        'category': category,
                        'content': f'Error downloading: HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                results.append({
                    'category': category,
                    'content': f'Error downloading: {str(e)}'
                })
                
        else:
            # Document not found
            available_keys = list(doc_lookup.keys())[:10]  # Show first 10 available keys
            results.append({
                'category': category,
                'content': f'Document not found: {category}_{index}. Available keys: {available_keys}'
            })
            print(f"✗ Document {category}_{index} not found")
    
    return results


if __name__ == "__main__":
    user_id = "test_user"
    print(f"Creating session for user: {user_id}")
    session = _create_session(user_id)
    session_id = session.get("session_id")
    print(f"Session ID: {session_id}")

    upload = asyncio.run(upload_folder_optimized(session_id, "lumir_data"))
    # print(f"Upload results: {upload}")

    # Original test (commented out for now)
    dict_test = get_documents_content_by_indicators(session_id, [
        {"category": "tai", "index": "1"},
        {"category": "ssi", "index": "3"},
        {"category": "ri", "index": "1"},
        {"category": "ami", "index": "2"},
    ])  # Should retrieve documents by category and index
    for item in dict_test:
        print(f"\nCategory: {item['category']}")
        print(f"Filename: {item.get('filename', 'N/A')}")
        print(f"Relative Path: {item.get('relative_path', 'N/A')}")
        print(f"File Size: {item.get('file_size', 'N/A')} bytes")
        print(f"Content Preview:\n{item['content'][:500]}")  # Print first 500 chars of content