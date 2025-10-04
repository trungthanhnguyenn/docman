"""
Embedding Service for DOCMAN

Adapted from lumir-agentic/module/document/embedding_manager.py
Provides server-side embedding generation for text chunks.

Features:
- Multiple providers (HuggingFace, Gemini, Sentence Transformers)
- CPU/GPU device detection (CPU default, GPU fallback)
- Model selection with Qwen3-0.6B default (1024 dim)
- Batch processing support
- Rate limiting for API-based embeddings
- Proper resource cleanup to prevent semaphore leaks
"""

import os
import time
import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import lifecycle manager
from src.api.services.model_lifecycle import (
    get_model_lifecycle_manager,
    cleanup_sentence_transformer,
    cleanup_hf_embeddings
)

# Embedding libraries with fallback support
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HuggingFaceEmbeddings = None
    HF_AVAILABLE = False
    logger.warning("langchain-huggingface not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, CPU-only mode")

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.warning("Gemini API not available")


class EmbeddingService:
    """
    Embedding service for generating text embeddings
    
    Supports:
    - HuggingFace models (default: Qwen3-Embedding-0.6B, 1024 dim)
    - Sentence Transformers
    - Google Gemini embeddings
    - CPU/GPU auto-detection with CPU default
    - Proper resource cleanup
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        provider: str = "hf",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize embedding service
        
        Args:
            model_name: Model identifier (default: Qwen3-Embedding-0.6B)
            provider: Provider type ('hf', 'sentence-transformers', 'gemini')
            batch_size: Batch size for embedding generation
            device: Device to use ('cpu', 'cuda', 'mps') - auto-detected if None
        """
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.model = None
        self.embedding_dimension = 1024  # Default for Qwen3-0.6B
        
        # Rate limiting for API-based providers
        self.rpm_limit = int(os.getenv("EMBEDDING_RPM_LIMIT", "80"))
        self._window_start = time.time()
        self._requests_in_window = 0
        
        # Gemini client
        self._gemini_client = None
        
        # Device detection
        self.device = device or self._detect_device()
        logger.info(f"Embedding service using device: {self.device}")
        
        # Lifecycle manager
        self._lifecycle_manager = get_model_lifecycle_manager()
        self._model_registered = False
        
        # Initialize model
        self._initialize_model()
    
    def _detect_device(self) -> str:
        """Detect optimal device (CPU default, GPU fallback if available)"""
        if not TORCH_AVAILABLE:
            logger.info("Torch not available, using CPU")
            return "cpu"
        
        try:
            import torch
            import platform
            
            # Check Apple Silicon MPS
            if platform.system() == "Darwin" and platform.machine() in ["arm64", "arm64e"]:
                if torch.backends.mps.is_available():
                    logger.info("Apple Silicon GPU (MPS) detected and available")
                    return "mps"
                else:
                    logger.info("Apple Silicon detected, using CPU")
                    return "cpu"
            
            # Check CUDA
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA detected with {gpu_count} GPU(s)")
                return "cuda"
            
            # Default to CPU
            logger.info("Using CPU for embeddings (default)")
            return "cpu"
            
        except Exception as e:
            logger.warning(f"Device detection failed, using CPU: {e}")
            return "cpu"
    
    def _initialize_model(self):
        """Initialize embedding model based on provider"""
        try:
            if self.provider == "hf" and HF_AVAILABLE:
                self._initialize_hf_model()
            elif self.provider == "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                self._initialize_st_model()
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                self._initialize_gemini()
            else:
                raise RuntimeError(
                    f"Provider '{self.provider}' not available. "
                    f"Available: HF={HF_AVAILABLE}, ST={SENTENCE_TRANSFORMERS_AVAILABLE}, Gemini={GEMINI_AVAILABLE}"
                )
            
            logger.info(f"✅ Embedding model initialized: {self.model_name} ({self.embedding_dimension}d)")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_hf_model(self):
        """Initialize HuggingFace embeddings model"""
        model_name = self.model_name
        if "/" not in model_name:
            model_name = f"Qwen/{model_name}"
        
        # Get HF token from environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "trust_remote_code": True,
                "device": self.device
            },
            encode_kwargs={
                "batch_size": self.batch_size
            }
        )
        
        # Probe embedding dimension
        probe_vec = self.model.embed_query("probe")
        if not isinstance(probe_vec, list) or len(probe_vec) == 0:
            raise RuntimeError("Embedding model returned invalid probe vector")
        
        self.embedding_dimension = len(probe_vec)
        logger.info(f"HF model loaded: {model_name}, dimension: {self.embedding_dimension}")
        
        # Register with lifecycle manager
        self._lifecycle_manager.register_model(
            name=f"embedding_hf_{id(self)}",
            model=self.model,
            cleanup_fn=cleanup_hf_embeddings
        )
        self._model_registered = True
    
    def _initialize_st_model(self):
        """Initialize Sentence Transformers model"""
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"SentenceTransformer loaded: {self.model_name}, dimension: {self.embedding_dimension}")
        
        # Register with lifecycle manager
        self._lifecycle_manager.register_model(
            name=f"embedding_st_{id(self)}",
            model=self.model,
            cleanup_fn=cleanup_sentence_transformer
        )
        self._model_registered = True
    
    def _initialize_gemini(self):
        """Initialize Gemini embeddings"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        try:
            self._gemini_client = google_genai.Client(api_key=api_key)
            self.model = "gemini-embedding-001"
            self.embedding_dimension = 768  # Gemini embedding dimension
            logger.info("Gemini embeddings initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _check_rate_limit(self):
        """Simple rate limiting for API calls"""
        current_time = time.time()
        if current_time - self._window_start > 60:
            # Reset window
            self._window_start = current_time
            self._requests_in_window = 0
        
        if self._requests_in_window >= self.rpm_limit:
            sleep_time = 60 - (current_time - self._window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._window_start = time.time()
                self._requests_in_window = 0
        
        self._requests_in_window += 1
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        try:
            if self.provider == "hf":
                return self.model.embed_query(text)
            
            elif self.provider == "sentence-transformers":
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            
            elif self.provider == "gemini":
                self._check_rate_limit()
                result = self._gemini_client.models.embed_content(
                    model=self.model,
                    content=text
                )
                return result.embeddings[0].values
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if self.provider == "hf":
                return self.model.embed_documents(texts)
            
            elif self.provider == "sentence-transformers":
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                return embeddings.tolist()
            
            elif self.provider == "gemini":
                # Batch with rate limiting
                results = []
                for text in texts:
                    self._check_rate_limit()
                    result = self._gemini_client.models.embed_content(
                        model=self.model,
                        content=text
                    )
                    results.append(result.embeddings[0].values)
                return results
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dimension
    
    def cleanup(self):
        """Cleanup resources"""
        if self._model_registered and self.model:
            model_name = f"embedding_{self.provider}_{id(self)}"
            self._lifecycle_manager.unregister_model(model_name)
            self._model_registered = False
            logger.info(f"✅ Embedding service cleaned up: {self.model_name}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup in destructor failed: {e}")


# Singleton instance for service-wide use
_embedding_service_instance: Optional[EmbeddingService] = None
_embedding_service_lock = threading.Lock()


def get_embedding_service(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    reinit: bool = False
) -> EmbeddingService:
    """
    Get or create singleton embedding service instance (thread-safe)
    
    Args:
        model_name: Model name (uses env or default if None)
        provider: Provider name (uses env or default if None)
        reinit: Force reinitialization
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service_instance
    
    # Thread-safe singleton pattern
    with _embedding_service_lock:
        if _embedding_service_instance is None or reinit:
            # Cleanup old instance if reinitializing
            if _embedding_service_instance and reinit:
                _embedding_service_instance.cleanup()
            
            model_name = model_name or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
            provider = provider or os.getenv("EMBEDDING_PROVIDER", "hf")
            
            logger.info(f"Initializing embedding service: {provider}/{model_name}")
            _embedding_service_instance = EmbeddingService(
                model_name=model_name,
                provider=provider
            )
    
    return _embedding_service_instance


def cleanup_embedding_service():
    """Cleanup the singleton embedding service"""
    global _embedding_service_instance
    
    with _embedding_service_lock:
        if _embedding_service_instance:
            _embedding_service_instance.cleanup()
            _embedding_service_instance = None
            logger.info("✅ Embedding service singleton cleaned up")
