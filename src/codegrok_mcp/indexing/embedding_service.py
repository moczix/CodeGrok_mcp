"""
Native embedding service using SentenceTransformers.

Default: Nomic Embed Code - SOTA code embedding model (81.7% Python, 80.5% Java on CodeSearchNet)

This replaces Ollama API calls with direct model execution for:
- 10-20x faster embedding generation
- True batch processing
- GPU acceleration support
- No external server dependency
- Code-optimized embeddings for better retrieval

Usage:
    from codegrok_mcp.indexing.embedding_service import EmbeddingService, get_embedding_service

    # Get singleton instance
    service = get_embedding_service()

    # Embed single text
    embedding = service.embed("def hello(): pass")

    # Embed batch (much faster)
    embeddings = service.embed_batch(["text1", "text2", "text3"])
"""

import gc
import threading
from functools import lru_cache
from typing import List, Optional, Union

# Lazy imports for optional dependencies
_sentence_transformers = None
_torch = None


def _import_dependencies():
    """Lazy import heavy dependencies."""
    global _sentence_transformers, _torch

    if _sentence_transformers is None:
        try:
            import sentence_transformers
            _sentence_transformers = sentence_transformers
        except ImportError:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for native embedding. "
                "Install with: pip install sentence-transformers"
            )

    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for native embedding. "
                "Install with: pip install torch"
            )

    return _sentence_transformers, _torch


# Supported embedding models with their configurations
EMBEDDING_MODELS = {
    # SOTA code embedding model - best for code analysis
    'nomic-embed-code': {
        'hf_name': 'nomic-ai/nomic-embed-code-v1',
        'dimensions': 768,
        'max_seq_length': 8192,
        'trust_remote_code': True,
        'prompt_prefix': 'search_document: ',
        'query_prefix': 'search_query: ',
    },
    # General text embedding (previous default)
    'nomic-embed-text': {
        'hf_name': 'nomic-ai/nomic-embed-text-v1.5',
        'dimensions': 768,
        'max_seq_length': 8192,
        'trust_remote_code': True,  # Required for nomic
        'prompt_prefix': 'search_document: ',  # Nomic uses task prefixes
        'query_prefix': 'search_query: ',
    },
    # Lightweight code embedding - efficient alternative
    'coderankembed': {
        'hf_name': 'nomic-ai/CodeRankEmbed-137M',
        'dimensions': 768,
        'max_seq_length': 8192,
        'trust_remote_code': True,
        'prompt_prefix': '',
        'query_prefix': '',
    },
    'bge-m3': {
        'hf_name': 'BAAI/bge-m3',
        'dimensions': 1024,
        'max_seq_length': 8192,
        'trust_remote_code': False,
        'prompt_prefix': '',
        'query_prefix': '',
    },
    'all-MiniLM-L6-v2': {
        'hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimensions': 384,
        'max_seq_length': 256,
        'trust_remote_code': False,
        'prompt_prefix': '',
        'query_prefix': '',
    },
    'all-mpnet-base-v2': {
        'hf_name': 'sentence-transformers/all-mpnet-base-v2',
        'dimensions': 768,
        'max_seq_length': 384,
        'trust_remote_code': False,
        'prompt_prefix': '',
        'query_prefix': '',
    },
}

# Default model - Nomic Embed Code (SOTA for code retrieval)
DEFAULT_MODEL = 'nomic-embed-code'

# Previous default - General text embedding (uncomment to swap back)
# DEFAULT_MODEL = 'nomic-embed-text'


class EmbeddingService:
    """
    Native embedding service using SentenceTransformers.

    Features:
    - Singleton pattern for consistency
    - Batch processing for efficiency
    - GPU acceleration when available
    - Memory management with GC triggers
    - Thread-safe operations
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_batch_size: int = 128,
        normalize: bool = True,
        show_progress: bool = False,  # Disabled to prevent tqdm output conflicts
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: Name of embedding model (see EMBEDDING_MODELS)
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Default batch size for encoding
            max_batch_size: Maximum allowed batch size
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar during encoding
            cache_dir: Directory to cache downloaded models
        """
        _import_dependencies()

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.normalize = normalize
        self.show_progress = show_progress

        # Get model config
        if model_name in EMBEDDING_MODELS:
            self.config = EMBEDDING_MODELS[model_name]
        else:
            # Assume it's a HuggingFace model name
            self.config = {
                'hf_name': model_name,
                'dimensions': None,  # Will be set after loading
                'max_seq_length': 512,
                'trust_remote_code': False,
                'prompt_prefix': '',
                'query_prefix': '',
            }

        # Determine device
        if device is None:
            device = 'cuda' if _torch.cuda.is_available() else 'cpu'  # pragma: no cover
        self.device = device

        # Thread safety
        self._lock = threading.Lock()
        self._model = None
        self._model_loaded = False

        # Stats
        self.stats = {
            'total_embeddings': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Cache directory
        self.cache_dir = cache_dir

        # Create cached embedding function (instance-level LRU cache)
        self._embed_cached = lru_cache(maxsize=1000)(self._embed_single_uncached)

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model_loaded:
            return

        with self._lock:
            if self._model_loaded:
                return

            if self.show_progress:
                print(f"Loading embedding model: {self.config['hf_name']}")
                print(f"Device: {self.device}")

            model_kwargs = {
                'device': self.device,
            }

            if self.cache_dir:
                model_kwargs['cache_folder'] = self.cache_dir

            if self.config.get('trust_remote_code'):
                model_kwargs['trust_remote_code'] = True

            # Suppress stdout/stderr during model loading to prevent
            # "<All keys matched successfully>" message from appearing.
            # This message comes from tqdm.write() during weight loading.
            import sys
            import io
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                self._model = _sentence_transformers.SentenceTransformer(
                    self.config['hf_name'],
                    **model_kwargs
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Update dimensions if not set
            if self.config['dimensions'] is None:
                self.config['dimensions'] = self._model.get_sentence_embedding_dimension()

            self._model_loaded = True
            if self.show_progress:
                print(f"Model loaded. Dimensions: {self.config['dimensions']}")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        self._load_model()
        return self.config['dimensions']

    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """
        Embed a single text without caching (internal use).

        Returns tuple for hashability in LRU cache.
        """
        self._load_model()

        # Add prefix if configured
        prefix = self.config['query_prefix'] if is_query else self.config['prompt_prefix']
        if prefix:
            text = prefix + text

        # Encode single text
        embeddings = self._model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        # Return as tuple for LRU cache hashability
        return tuple(embeddings[0].tolist())

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embed a single text with LRU caching.

        Repeated queries return instantly from cache (~0ms vs ~50ms).
        Cache size: 1000 entries (~3MB memory overhead).

        Args:
            text: Text to embed
            is_query: If True, use query prefix (for search queries)

        Returns:
            Embedding as list of floats
        """
        # Check if this will be a cache hit (for stats)
        cache_info_before = self._embed_cached.cache_info()

        # Get embedding (cached or computed)
        embedding_tuple = self._embed_cached(text, is_query)

        # Update stats
        cache_info_after = self._embed_cached.cache_info()
        if cache_info_after.hits > cache_info_before.hits:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
            self.stats['total_embeddings'] += 1

        return list(embedding_tuple)

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of texts to embed
            is_query: If True, use query prefix (for search queries)
            batch_size: Override default batch size

        Returns:
            List of embeddings as lists of floats
        """
        import time

        if not texts:
            return []

        self._load_model()

        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, self.max_batch_size)

        # Add prefix if configured
        prefix = self.config['query_prefix'] if is_query else self.config['prompt_prefix']
        if prefix:
            texts = [prefix + t for t in texts]

        start_time = time.time()

        # Encode with SentenceTransformers
        # Note: show_progress_bar=False to prevent tqdm from intercepting stdout
        # which causes "<All keys matched successfully>" messages to appear
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,  # We use our own progress bar in display.py
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        elapsed = time.time() - start_time

        # Update stats
        self.stats['total_embeddings'] += len(texts)
        self.stats['total_batches'] += (len(texts) + batch_size - 1) // batch_size
        self.stats['total_time'] += elapsed

        # Memory management - trigger GC periodically
        if self.stats['total_batches'] % 100 == 0:
            gc.collect()
            if self.device == 'cuda':  # pragma: no cover
                _torch.cuda.empty_cache()

        # Convert to list of lists (ChromaDB compatible)
        return embeddings.tolist()

    def get_stats(self) -> dict:
        """Get embedding statistics."""
        stats = self.stats.copy()
        if stats['total_time'] > 0:
            stats['embeddings_per_second'] = stats['total_embeddings'] / stats['total_time']
        else:
            stats['embeddings_per_second'] = 0
        return stats

    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        cache_info = self._embed_cached.cache_info()
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        return {
            'hits': self.stats['cache_hits'],
            'misses': self.stats['cache_misses'],
            'hit_rate': f"{hit_rate:.1%}",
            'size': cache_info.currsize,
            'maxsize': cache_info.maxsize,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embed_cached.cache_clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_embeddings': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def unload(self):
        """Unload model to free memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._model_loaded = False
                self._embed_cached.cache_clear()  # Clear embedding cache
                gc.collect()
                if self.device == 'cuda':  # pragma: no cover
                    _torch.cuda.empty_cache()


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None
_singleton_lock = threading.Lock()


def get_embedding_service(
    model_name: str = DEFAULT_MODEL,
    **kwargs
) -> EmbeddingService:
    """
    Get singleton embedding service instance.

    This ensures all embeddings use the same model instance,
    preventing embedding space inconsistencies.

    Args:
        model_name: Embedding model name
        **kwargs: Additional arguments for EmbeddingService

    Returns:
        Singleton EmbeddingService instance
    """
    global _embedding_service

    if _embedding_service is None:
        with _singleton_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService(model_name, **kwargs)

    return _embedding_service


def reset_embedding_service():
    """Reset the singleton instance (useful for testing)."""
    global _embedding_service

    with _singleton_lock:
        if _embedding_service is not None:
            _embedding_service.unload()
            _embedding_service = None


class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB that uses our native service.

    Usage with ChromaDB:
        collection = client.create_collection(
            name="codebase",
            embedding_function=ChromaDBEmbeddingFunction()
        )
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._service = None

    @property
    def service(self) -> EmbeddingService:
        if self._service is None:
            self._service = get_embedding_service(self.model_name)
        return self._service

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for ChromaDB."""
        return self.service.embed_batch(input, is_query=False)


# Convenience function for quick embedding
def embed(texts: Union[str, List[str]], is_query: bool = False) -> Union[List[float], List[List[float]]]:
    """
    Quick embedding function.

    Args:
        texts: Single text or list of texts
        is_query: If True, use query prefix

    Returns:
        Single embedding or list of embeddings
    """
    service = get_embedding_service()

    if isinstance(texts, str):
        return service.embed(texts, is_query=is_query)
    else:
        return service.embed_batch(texts, is_query=is_query)
