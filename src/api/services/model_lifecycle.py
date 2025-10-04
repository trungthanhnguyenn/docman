"""
ML Model Lifecycle Manager

Quản lý lifecycle của các ML models để tránh resource leaks.
Đảm bảo proper cleanup khi application shutdown.
"""

import logging
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """
    Singleton manager for ML model lifecycle
    
    Features:
    - Thread-safe initialization
    - Proper cleanup on shutdown
    - Resource tracking
    - Graceful degradation
    """
    
    _instance: Optional['ModelLifecycleManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._models: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._shutdown_hooks = []
        
        logger.info("✅ ModelLifecycleManager initialized")
    
    def register_model(self, name: str, model: Any, cleanup_fn: Optional[callable] = None):
        """
        Register a model for lifecycle management
        
        Args:
            name: Unique model identifier
            model: The model instance
            cleanup_fn: Optional cleanup function to call on shutdown
        """
        with self._lock:
            if name in self._models:
                logger.warning(f"Model {name} already registered, replacing")
                self.unregister_model(name)
            
            self._models[name] = model
            self._model_locks[name] = threading.Lock()
            
            if cleanup_fn:
                self._shutdown_hooks.append((name, cleanup_fn))
            
            logger.info(f"✅ Registered model: {name}")
    
    def unregister_model(self, name: str):
        """Unregister and cleanup a model"""
        with self._lock:
            if name not in self._models:
                logger.warning(f"Model {name} not registered")
                return
            
            # Run cleanup hooks
            for hook_name, cleanup_fn in self._shutdown_hooks:
                if hook_name == name:
                    try:
                        logger.info(f"🧹 Running cleanup for {name}")
                        cleanup_fn(self._models[name])
                    except Exception as e:
                        logger.error(f"❌ Cleanup failed for {name}: {e}")
            
            # Remove from tracking
            del self._models[name]
            del self._model_locks[name]
            self._shutdown_hooks = [(n, f) for n, f in self._shutdown_hooks if n != name]
            
            logger.info(f"✅ Unregistered model: {name}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model (thread-safe)"""
        with self._lock:
            return self._models.get(name)
    
    def get_model_lock(self, name: str) -> Optional[threading.Lock]:
        """Get the lock for a model"""
        return self._model_locks.get(name)
    
    @contextmanager
    def use_model(self, name: str):
        """
        Context manager for thread-safe model usage
        
        Example:
            with manager.use_model('embedding') as model:
                embeddings = model.encode(texts)
        """
        lock = self.get_model_lock(name)
        if lock:
            with lock:
                yield self.get_model(name)
        else:
            yield self.get_model(name)
    
    def cleanup_all(self):
        """Cleanup all registered models"""
        logger.info("🧹 Starting cleanup of all models...")
        
        with self._lock:
            model_names = list(self._models.keys())
        
        for name in model_names:
            self.unregister_model(name)
        
        logger.info("✅ All models cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about registered models"""
        with self._lock:
            return {
                "total_models": len(self._models),
                "models": list(self._models.keys()),
                "shutdown_hooks": len(self._shutdown_hooks)
            }


# Global instance
def get_model_lifecycle_manager() -> ModelLifecycleManager:
    """Get the singleton ModelLifecycleManager instance"""
    return ModelLifecycleManager()


# Cleanup functions for different model types
def cleanup_sentence_transformer(model):
    """Cleanup SentenceTransformer model"""
    try:
        # SentenceTransformer doesn't have explicit cleanup
        # but we can clear caches
        if hasattr(model, '_modules'):
            model._modules.clear()
        logger.info("SentenceTransformer cleanup completed")
    except Exception as e:
        logger.warning(f"SentenceTransformer cleanup warning: {e}")


def cleanup_cross_encoder(model):
    """Cleanup CrossEncoder model"""
    try:
        # CrossEncoder cleanup
        if hasattr(model, 'model'):
            del model.model
        logger.info("CrossEncoder cleanup completed")
    except Exception as e:
        logger.warning(f"CrossEncoder cleanup warning: {e}")


def cleanup_hf_embeddings(model):
    """Cleanup HuggingFaceEmbeddings model"""
    try:
        # HuggingFace cleanup
        if hasattr(model, 'client'):
            del model.client
        logger.info("HuggingFaceEmbeddings cleanup completed")
    except Exception as e:
        logger.warning(f"HuggingFaceEmbeddings cleanup warning: {e}")
