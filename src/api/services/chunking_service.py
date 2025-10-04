"""
Document Chunking Service

Provides intelligent document chunking with multiple strategies:
- auto: Automatically detect best strategy
- header_based: Chunk by document headers
- qa_based: Chunk by Q&A pairs
- semantic: Chunk by semantic boundaries
- size_based: Fixed-size chunks
- advanced_semantic: Use sentence transformers for semantic clustering

This service is inspired by lumir-agentic document_processor but adapted for DocMan.
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available. Advanced semantic chunking disabled.")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("Language detection not available. Using default language.")


@dataclass
class ChunkResult:
    """Chunk result data structure"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_type: str  # 'header', 'content', 'qa_pair', 'list_item', 'semantic'
    metadata: Dict[str, Any]
    token_count: int
    language: str = "vi"


@dataclass
class ChunkingResponse:
    """Complete chunking response"""
    chunks: List[ChunkResult]
    total_chunks: int
    strategy_used: str
    processing_time_ms: int
    document_metadata: Dict[str, Any]


class ChunkingConfig:
    """Chunking configuration"""
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000,
        semantic_threshold: float = 0.3
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.semantic_threshold = semantic_threshold


class DocumentChunkingService:
    """
    Main document chunking service
    Provides multiple chunking strategies with automatic detection
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.semantic_model = None
        self.embedding_cache = {}
        
        # Initialize semantic model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Semantic chunking model loaded: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
    
    def chunk_content(
        self,
        content: str,
        chunk_mode: str = "auto",
        filename: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> ChunkingResponse:
        """
        Main chunking entry point
        
        Args:
            content: Document content to chunk
            chunk_mode: Chunking strategy (auto, header_based, qa_based, semantic, size_based, advanced_semantic)
            filename: Optional filename for context
            document_id: Optional document ID
            
        Returns:
            ChunkingResponse with chunks and metadata
            
        Raises:
            ValueError: If content is empty or invalid mode
        """
        start_time = time.time()
        
        # Validate input
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        if chunk_mode not in ["auto", "header_based", "qa_based", "semantic", "size_based", "advanced_semantic"]:
            raise ValueError(
                f"Invalid chunk_mode: {chunk_mode}. "
                f"Must be one of: auto, header_based, qa_based, semantic, size_based, advanced_semantic"
            )
        
        logger.info(f"Starting chunking with mode: {chunk_mode}")
        
        # Analyze document structure
        structure = self._analyze_document_structure(content)
        
        # Determine strategy
        if chunk_mode == "auto":
            strategy = self._determine_optimal_strategy(structure)
            logger.info(f"Auto-detected strategy: {strategy}")
        else:
            strategy = chunk_mode
        
        # Execute chunking strategy
        try:
            if strategy == "header_based":
                chunks = self._chunk_by_headers(content, structure)
            elif strategy == "qa_based":
                chunks = self._chunk_by_qa_pairs(content, structure)
            elif strategy == "semantic":
                chunks = self._chunk_by_semantic_boundaries(content, structure)
            elif strategy == "advanced_semantic":
                if self.semantic_model:
                    chunks = self._chunk_by_advanced_semantic(content, structure)
                else:
                    logger.warning("Advanced semantic chunking not available, falling back to semantic")
                    chunks = self._chunk_by_semantic_boundaries(content, structure)
            else:  # size_based
                chunks = self._chunk_by_size(content)
        except Exception as e:
            logger.error(f"Chunking failed with strategy {strategy}: {e}")
            raise ValueError(f"Chunking failed: {str(e)}")
        
        # Validate chunks
        if not chunks:
            raise ValueError("Chunking produced no results")
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        document_metadata = {
            "filename": filename,
            "document_id": document_id,
            "content_length": len(content),
            "detected_language": structure.get("detected_language", "vi"),
            "structure_analysis": {
                "total_lines": structure["total_lines"],
                "headers_count": len(structure.get("headers", [])),
                "qa_pairs_count": len(structure.get("qa_pairs", [])),
                "lists_count": len(structure.get("lists", []))
            }
        }
        
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=strategy,
            processing_time_ms=processing_time_ms,
            document_metadata=document_metadata
        )
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure to determine optimal chunking strategy"""
        lines = content.split('\n')
        structure = {
            "total_lines": len(lines),
            "total_chars": len(content),
            "detected_language": self._detect_language(content),
            "headers": [],
            "qa_pairs": [],
            "lists": []
        }
        
        # Header detection patterns
        header_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # 1. Title
            r'^\d+\.\d+\s+[A-Z]', # 1.1. Subtitle
            r'^[A-Z][^.!?]*:$',   # Title ending with :
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for headers
            for pattern in header_patterns:
                if re.match(pattern, line):
                    header_level = self._determine_header_level(line)
                    structure["headers"].append({
                        "text": line,
                        "line_number": i,
                        "level": header_level
                    })
                    break
            
            # Detect Q&A pairs
            if '?' in line and len(line) < 200:
                if i + 1 < len(lines) and lines[i + 1].strip():
                    structure["qa_pairs"].append({
                        "question": line,
                        "answer": lines[i + 1].strip(),
                        "line_number": i
                    })
            
            # Detect lists
            if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                structure["lists"].append({
                    "text": line,
                    "line_number": i
                })
        
        return structure
    
    def _determine_optimal_strategy(self, structure: Dict[str, Any]) -> str:
        """Determine the best chunking strategy based on document structure"""
        total_lines = structure["total_lines"]
        
        if total_lines == 0:
            return "size_based"
        
        qa_ratio = len(structure["qa_pairs"]) / total_lines
        header_ratio = len(structure["headers"]) / total_lines
        list_ratio = len(structure["lists"]) / total_lines
        
        # Decision tree
        if qa_ratio > 0.1:  # >10% is Q&A
            return "qa_based"
        elif header_ratio > 0.05:  # >5% is headers
            return "header_based"
        elif list_ratio > 0.15 or total_lines > 100:  # >15% is lists or long document
            if self.semantic_model:
                return "advanced_semantic"
            else:
                return "semantic"
        else:
            return "size_based"
    
    def _chunk_by_headers(self, content: str, structure: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk document by headers with small chunk merging"""
        lines = content.split('\n')
        headers = structure.get("headers", [])
        
        if not headers:
            logger.warning("No headers found, falling back to size-based chunking")
            return self._chunk_by_size(content)
        
        headers.sort(key=lambda x: x["line_number"])
        
        # First pass: create all chunks
        temp_chunks = []
        for i, header in enumerate(headers):
            start_line = header["line_number"]
            end_line = headers[i + 1]["line_number"] if i + 1 < len(headers) else len(lines)
            
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines).strip()
            
            if chunk_content:
                start_char = len("\n".join(lines[:start_line])) + (1 if start_line > 0 else 0)
                end_char = len("\n".join(lines[:end_line]))
                
                temp_chunks.append({
                    "content": chunk_content,
                    "header": header,
                    "start_char": start_char,
                    "end_char": end_char,
                    "size": len(chunk_content)
                })
        
        # Second pass: merge small chunks
        merged_chunks = []
        i = 0
        while i < len(temp_chunks):
            current = temp_chunks[i]
            
            if current["size"] < self.config.min_chunk_size:
                if i + 1 < len(temp_chunks):
                    # Merge with next
                    next_chunk = temp_chunks[i + 1]
                    merged_content = current["content"] + "\n\n" + next_chunk["content"]
                    merged_chunks.append({
                        "content": merged_content,
                        "header": current["header"],
                        "start_char": current["start_char"],
                        "end_char": next_chunk["end_char"],
                        "size": len(merged_content)
                    })
                    i += 2
                elif merged_chunks:
                    # Merge with previous
                    prev_chunk = merged_chunks.pop()
                    merged_content = prev_chunk["content"] + "\n\n" + current["content"]
                    merged_chunks.append({
                        "content": merged_content,
                        "header": prev_chunk["header"],
                        "start_char": prev_chunk["start_char"],
                        "end_char": current["end_char"],
                        "size": len(merged_content)
                    })
                    i += 1
                else:
                    merged_chunks.append(current)
                    i += 1
            else:
                merged_chunks.append(current)
                i += 1
        
        # Create final chunks
        result_chunks = []
        for idx, chunk_data in enumerate(merged_chunks):
            result_chunks.append(ChunkResult(
                content=chunk_data["content"],
                chunk_index=idx,
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                chunk_type="header",
                metadata={
                    "header_text": chunk_data["header"]["text"],
                    "header_level": chunk_data["header"]["level"],
                    "strategy": "header_based"
                },
                token_count=len(chunk_data["content"].split()),
                language=structure.get("detected_language", "vi")
            ))
        
        return result_chunks
    
    def _chunk_by_qa_pairs(self, content: str, structure: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk document by Q&A pairs"""
        qa_pairs = structure.get("qa_pairs", [])
        
        if not qa_pairs:
            logger.warning("No Q&A pairs found, falling back to size-based chunking")
            return self._chunk_by_size(content)
        
        lines = content.split('\n')
        chunks = []
        chunk_index = 0
        
        # Track Q&A lines
        qa_lines = set()
        for qa in qa_pairs:
            line_num = qa["line_number"]
            qa_lines.add(line_num)
            qa_lines.add(line_num + 1)
        
        # Create Q&A chunks
        for qa_pair in qa_pairs:
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            qa_content = f"Q: {question}\nA: {answer}"
            
            start_pos = content.find(question)
            if start_pos != -1:
                end_pos = start_pos + len(qa_content)
                
                chunks.append(ChunkResult(
                    content=qa_content,
                    chunk_index=chunk_index,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_type="qa_pair",
                    metadata={
                        "question": question,
                        "answer": answer,
                        "strategy": "qa_based"
                    },
                    token_count=len(qa_content.split()),
                    language=structure.get("detected_language", "vi")
                ))
                chunk_index += 1
        
        # Capture non-Q&A content
        non_qa_content = []
        current_text = []
        
        for i, line in enumerate(lines):
            if i not in qa_lines:
                line = line.strip()
                if line:
                    current_text.append(line)
            else:
                if current_text:
                    text = "\n".join(current_text)
                    if len(text) > 30:
                        non_qa_content.append(text)
                    current_text = []
        
        if current_text:
            text = "\n".join(current_text)
            if len(text) > 30:
                non_qa_content.append(text)
        
        # Add non-Q&A content as chunks
        for text in non_qa_content:
            start_pos = content.find(text)
            if start_pos != -1:
                chunks.append(ChunkResult(
                    content=text,
                    chunk_index=chunk_index,
                    start_char=start_pos,
                    end_char=start_pos + len(text),
                    chunk_type="content",
                    metadata={"strategy": "qa_based", "type": "non_qa_content"},
                    token_count=len(text.split()),
                    language=structure.get("detected_language", "vi")
                ))
                chunk_index += 1
        
        return chunks
    
    def _chunk_by_semantic_boundaries(self, content: str, structure: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk by semantic boundaries (paragraphs)"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        chunk_index = 0
        current_chunk = ""
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if can add paragraph to current chunk
            if len(current_chunk) + len(paragraph) <= self.config.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk
                if current_chunk:
                    end_pos = start_pos + len(current_chunk)
                    chunks.append(ChunkResult(
                        content=current_chunk,
                        chunk_index=chunk_index,
                        start_char=start_pos,
                        end_char=end_pos,
                        chunk_type="semantic",
                        metadata={
                            "strategy": "semantic",
                            "paragraph_count": current_chunk.count('\n\n') + 1
                        },
                        token_count=len(current_chunk.split()),
                        language=structure.get("detected_language", "vi")
                    ))
                    chunk_index += 1
                    start_pos = end_pos
                
                # Start new chunk
                current_chunk = paragraph
        
        # Save last chunk
        if current_chunk:
            end_pos = start_pos + len(current_chunk)
            chunks.append(ChunkResult(
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=start_pos,
                end_char=end_pos,
                chunk_type="semantic",
                metadata={
                    "strategy": "semantic",
                    "paragraph_count": current_chunk.count('\n\n') + 1
                },
                token_count=len(current_chunk.split()),
                language=structure.get("detected_language", "vi")
            ))
        
        return chunks
    
    def _chunk_by_size(self, content: str) -> List[ChunkResult]:
        """Fixed-size chunking with overlap"""
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = start_pos + self.config.chunk_size
            
            # Find suitable cut position (sentence boundary)
            if end_pos < len(content):
                for sep in ['. ', '! ', '? ', '\n']:
                    pos = content.rfind(sep, start_pos, end_pos)
                    if pos != -1:
                        end_pos = pos + len(sep)
                        break
            
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:
                chunks.append(ChunkResult(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_type="content",
                    metadata={"strategy": "size_based"},
                    token_count=len(chunk_content.split()),
                    language="vi"
                ))
                chunk_index += 1
            
            # Move to next chunk with overlap
            start_pos = end_pos - self.config.chunk_overlap if end_pos < len(content) else end_pos
        
        return chunks
    
    def _chunk_by_advanced_semantic(self, content: str, structure: Dict[str, Any]) -> List[ChunkResult]:
        """Advanced semantic chunking using sentence transformers"""
        if not self.semantic_model:
            raise ValueError("Semantic model not available")
        
        # Extract sentences
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        
        if len(sentences) < 2:
            return self._chunk_by_size(content)
        
        # Generate embeddings
        embeddings = self.semantic_model.encode(sentences)
        
        # Cluster sentences
        n_clusters = max(2, min(len(sentences) // 3, 10))
        similarity_matrix = cosine_similarity(embeddings)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(1 - similarity_matrix)
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])
        
        # Create chunks from clusters
        chunks = []
        chunk_index = 0
        
        for cluster_id in sorted(clusters.keys()):
            cluster_content = " ".join(clusters[cluster_id])
            start_pos = content.find(clusters[cluster_id][0])
            end_pos = start_pos + len(cluster_content)
            
            chunks.append(ChunkResult(
                content=cluster_content,
                chunk_index=chunk_index,
                start_char=start_pos,
                end_char=end_pos,
                chunk_type="semantic",
                metadata={
                    "strategy": "advanced_semantic",
                    "cluster_id": int(cluster_id),
                    "sentence_count": len(clusters[cluster_id])
                },
                token_count=len(cluster_content.split()),
                language=structure.get("detected_language", "vi")
            ))
            chunk_index += 1
        
        return chunks
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        if LANGDETECT_AVAILABLE:
            try:
                sample = text[:min(1000, len(text))]
                return detect(sample)
            except LangDetectException:
                return "vi"
        return "vi"
    
    def _determine_header_level(self, header_text: str) -> int:
        """Determine header level"""
        if re.match(r'^\d+\.\d+\.\d+', header_text):
            return 3
        elif re.match(r'^\d+\.\d+', header_text):
            return 2
        elif re.match(r'^\d+\.', header_text):
            return 1
        elif header_text.isupper() and len(header_text) > 3:
            return 1
        return 0


# Singleton instance
_chunking_service: Optional[DocumentChunkingService] = None


def get_chunking_service(config: Optional[ChunkingConfig] = None) -> DocumentChunkingService:
    """Get or create chunking service singleton"""
    global _chunking_service
    
    if _chunking_service is None or config is not None:
        _chunking_service = DocumentChunkingService(config)
    
    return _chunking_service
