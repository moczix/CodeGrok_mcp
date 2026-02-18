"""
Embedding service supporting multiple providers: local (SentenceTransformers) and Ollama.

Default: Native embedding with Nomic Embed Code - SOTA code embedding model
(81.7% Python, 80.5% Java on CodeSearchNet)

Alternative: Ollama API for remote embedding generation.

Configuration via environment variables:
    CODEGROK_EMBEDDING_PROVIDER=local|ollama  (default: local)
    CODEGROK_OLLAMA_URL=http://localhost:11434  (default: http://localhost:11434)
    CODEGROK_OLLAMA_MODEL=nomic-embed-text  (required for ollama provider)
    CODEGROK_OLLAMA_DIMENSIONS=768  (required for ollama provider)

Usage:
    from codegrok_mcp.indexing.embedding_service import get_embedding_service

    # Get service instance (auto-detects provider from env)
    service = get_embedding_service()

    # Embed single text
    embedding = service.embed("def hello(): pass")

    # Embed batch
    embeddings = service.embed_batch(["text1", "text2", "text3"])
"""

import gc
import os
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Union

# Supported embedding models with their configurations
EMBEDDING_MODELS = {
    # Default: Lightweight code embedding - efficient (137M params, ~521MB)
    # SOTA on CodeSearchNet for its size class
    "coderankembed": {
        "hf_name": "nomic-ai/CodeRankEmbed",
        "dimensions": 768,
        "max_seq_length": 8192,
        "trust_remote_code": True,
        "prompt_prefix": "",
        "query_prefix": "Represent this query for searching relevant code: ",
    },
    # Example template - copy this to add your own model
    "my-new-model": {
        "hf_name": "organization/model-name",  # HuggingFace model ID
        "dimensions": 768,  # Output vector dimensions
        "max_seq_length": 512,  # Max input tokens
        "trust_remote_code": False,  # True if model needs custom code
        "prompt_prefix": "",  # Prepended to documents
        "query_prefix": "",  # Prepended to queries
    },
}

# Default model - CodeRankEmbed (137M params, SOTA for size, code-optimized)
DEFAULT_MODEL = "coderankembed"
DEFAULT_PROVIDER = "local"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class EmbeddingServiceBase(ABC):
    """
    Abstract base class for embedding services.

    Implementations:
    - NativeEmbeddingService: Uses SentenceTransformers (local)
    - OllamaEmbeddingService: Uses Ollama API (remote)
    """

    def __init__(self, dimensions: int, batch_size: int = 32, show_progress: bool = False):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._lock = threading.Lock()

        # Stats
        self.stats = {
            "total_embeddings": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Create cached embedding function (instance-level LRU cache)
        self._embed_cached = lru_cache(maxsize=1000)(self._embed_single_uncached)

    @abstractmethod
    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """
        Embed a single text without caching (internal use).

        Returns tuple for hashability in LRU cache.
        """
        pass

    @abstractmethod
    def embed_batch(
        self, texts: List[str], is_query: bool = False, batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed
            is_query: If True, use query prefix (for search queries)
            batch_size: Override default batch size

        Returns:
            List of embeddings as lists of floats
        """
        pass

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embed a single text with LRU caching.

        Args:
            text: Text to embed
            is_query: If True, use query prefix

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
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
            self.stats["total_embeddings"] += 1

        return list(embedding_tuple)

    def get_stats(self) -> dict:
        """Get embedding statistics."""
        stats = self.stats.copy()
        if stats["total_time"] > 0:
            stats["embeddings_per_second"] = stats["total_embeddings"] / stats["total_time"]
        else:
            stats["embeddings_per_second"] = 0
        return stats

    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        cache_info = self._embed_cached.cache_info()
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self.stats["cache_hits"],
            "misses": self.stats["cache_misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embed_cached.cache_clear()
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_embeddings": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @abstractmethod
    def unload(self):
        """Unload model to free memory."""
        pass


# Lazy imports for optional dependencies
_sentence_transformers = None
_torch = None


def _import_sentence_transformers():
    """Lazy import sentence-transformers."""
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
                "PyTorch is required for native embedding. " "Install with: pip install torch"
            )

    return _sentence_transformers, _torch


class NativeEmbeddingService(EmbeddingServiceBase):
    """
    Native embedding service using SentenceTransformers.

    Features:
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
        show_progress: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize native embedding service.

        Args:
            model_name: Name of embedding model (see EMBEDDING_MODELS)
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Default batch size for encoding
            max_batch_size: Maximum allowed batch size
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar during encoding
            cache_dir: Directory to cache downloaded models
        """
        _import_sentence_transformers()

        # Get model config
        if model_name in EMBEDDING_MODELS:
            self.config = EMBEDDING_MODELS[model_name]
        else:
            # Assume it's a HuggingFace model name
            self.config = {
                "hf_name": model_name,
                "dimensions": None,  # Will be set after loading
                "max_seq_length": 512,
                "trust_remote_code": False,
                "prompt_prefix": "",
                "query_prefix": "",
            }

        # Determine device
        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"  # pragma: no cover
        self.device = device
        self.max_batch_size = max_batch_size
        self.normalize = normalize
        self.cache_dir = cache_dir
        self._model = None
        self._model_loaded = False

        # Initialize base with dimensions (will be updated after model load)
        super().__init__(
            dimensions=self.config.get("dimensions", 768),
            batch_size=batch_size,
            show_progress=show_progress,
        )

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
                "device": self.device,
            }

            if self.cache_dir:
                model_kwargs["cache_folder"] = self.cache_dir

            if self.config.get("trust_remote_code"):
                model_kwargs["trust_remote_code"] = True

            # Suppress stdout/stderr during model loading
            import sys
            import io

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                self._model = _sentence_transformers.SentenceTransformer(
                    self.config["hf_name"], **model_kwargs
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Update dimensions if not set
            if self.config["dimensions"] is None:
                self.config["dimensions"] = self._model.get_sentence_embedding_dimension()
                self.dimensions = self.config["dimensions"]

            self._model_loaded = True
            if self.show_progress:
                print(f"Model loaded. Dimensions: {self.dimensions}")

    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """Embed a single text without caching."""
        self._load_model()

        # Add prefix if configured
        prefix = self.config["query_prefix"] if is_query else self.config["prompt_prefix"]
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

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts efficiently."""
        import time

        if not texts:
            return []

        self._load_model()

        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, self.max_batch_size)

        # Add prefix if configured
        prefix = self.config["query_prefix"] if is_query else self.config["prompt_prefix"]
        if prefix:
            texts = [prefix + t for t in texts]

        start_time = time.time()

        # Encode with SentenceTransformers
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        elapsed = time.time() - start_time

        # Update stats
        self.stats["total_embeddings"] += len(texts)
        self.stats["total_batches"] += (len(texts) + batch_size - 1) // batch_size
        self.stats["total_time"] += elapsed

        # Memory management
        if self.stats["total_batches"] % 100 == 0:
            gc.collect()
            if self.device == "cuda":  # pragma: no cover
                _torch.cuda.empty_cache()

        return embeddings.tolist()

    def unload(self):
        """Unload model to free memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._model_loaded = False
                self._embed_cached.cache_clear()
                gc.collect()
                if self.device == "cuda":  # pragma: no cover
                    _torch.cuda.empty_cache()


class OllamaEmbeddingService(EmbeddingServiceBase):
    """
    Ollama embedding service using Ollama API.

    Features:
    - Remote embedding generation via Ollama
    - Compatible with any Ollama embedding model
    - Batch processing with configurable size
    - Automatic retry on failure
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: int = 60,
        show_progress: bool = False,
    ):
        """
        Initialize Ollama embedding service.

        Args:
            base_url: Ollama API base URL
            model: Model name to use for embeddings
            dimensions: Embedding dimensions (must match model)
            batch_size: Batch size for requests
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
            show_progress: Show progress messages
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Try to import requests
        try:
            import requests

            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests is required for Ollama embedding. " "Install with: pip install requests"
            )

        super().__init__(dimensions=dimensions, batch_size=batch_size, show_progress=show_progress)

    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """Embed a single text via Ollama API."""
        # Ollama doesn't distinguish between query and document prefixes
        # We'll just embed as-is
        embedding = self._embed_with_retry(text)
        return tuple(embedding)

    def _embed_with_retry(self, text: str) -> List[float]:
        """Embed text with retry logic."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        for attempt in range(self.max_retries):
            try:
                response = self._requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                return result["embedding"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to get embedding from Ollama after {self.max_retries} attempts: {e}"
                    )
                import time

                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts via Ollama API."""
        import time

        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        embeddings = []

        start_time = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Ollama API processes one text at a time
            # We'll parallelize with threading for better performance
            batch_embeddings = self._embed_batch_parallel(batch)
            embeddings.extend(batch_embeddings)

            if self.show_progress and i % (batch_size * 10) == 0:
                progress = (i + len(batch)) / len(texts) * 100
                print(f"Embedding progress: {progress:.1f}%", end="\r")

        elapsed = time.time() - start_time

        # Update stats
        self.stats["total_embeddings"] += len(texts)
        self.stats["total_batches"] += (len(texts) + batch_size - 1) // batch_size
        self.stats["total_time"] += elapsed

        return embeddings

    def _embed_batch_parallel(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts in parallel using threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=min(len(texts), 8)) as executor:
            future_to_idx = {
                executor.submit(self._embed_with_retry, text): idx for idx, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to embed text at index {idx}: {e}")

        return embeddings

    def unload(self):
        """Unload - no-op for Ollama as model is server-side."""
        self._embed_cached.cache_clear()


def create_embedding_service(provider: Optional[str] = None, **kwargs) -> EmbeddingServiceBase:
    """
    Create an embedding service based on provider configuration.

    Args:
        provider: 'local', 'ollama', or None (auto-detect from env)
        **kwargs: Additional arguments passed to the service constructor

    Returns:
        EmbeddingServiceBase instance
    """
    # Determine provider from env or argument
    if provider is None:
        provider = os.environ.get("CODEGROK_EMBEDDING_PROVIDER", DEFAULT_PROVIDER)

    provider = provider.lower()

    if provider == "ollama":
        # Get Ollama configuration from environment
        base_url = os.environ.get("CODEGROK_OLLAMA_URL", DEFAULT_OLLAMA_URL)
        model = os.environ.get("CODEGROK_OLLAMA_MODEL", "nomic-embed-text")

        # Dimensions is required
        dimensions_str = os.environ.get("CODEGROK_OLLAMA_DIMENSIONS")
        if dimensions_str:
            dimensions = int(dimensions_str)
        else:
            raise ValueError(
                "CODEGROK_OLLAMA_DIMENSIONS environment variable is required when using Ollama provider. "
                "Example: CODEGROK_OLLAMA_DIMENSIONS=768"
            )

        return OllamaEmbeddingService(
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["model_name", "device", "normalize", "cache_dir"]
            },
        )

    elif provider == "local":
        model_name = kwargs.get("model_name", DEFAULT_MODEL)
        return NativeEmbeddingService(
            model_name=model_name,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["base_url", "model", "dimensions", "max_retries", "timeout"]
            },
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'local' or 'ollama'.")


# Model-keyed cache of embedding services
_embedding_services: Dict[str, EmbeddingServiceBase] = {}
_singleton_lock = threading.Lock()


def get_embedding_service(model_name: str = DEFAULT_MODEL, **kwargs) -> EmbeddingServiceBase:
    """
    Get embedding service instance.

    This uses a provider-keyed cache to ensure consistency.

    Args:
        model_name: Embedding model name (for local provider)
        **kwargs: Additional arguments for the service

    Returns:
        EmbeddingServiceBase instance
    """
    global _embedding_services

    # Determine provider
    provider = os.environ.get("CODEGROK_EMBEDDING_PROVIDER", DEFAULT_PROVIDER)

    # Create cache key based on provider and model
    if provider == "ollama":
        ollama_model = os.environ.get("CODEGROK_OLLAMA_MODEL", "nomic-embed-text")
        cache_key = f"ollama:{ollama_model}"
    else:
        cache_key = f"local:{model_name}"

    if cache_key not in _embedding_services:
        with _singleton_lock:
            if cache_key not in _embedding_services:
                _embedding_services[cache_key] = create_embedding_service(
                    provider=provider, model_name=model_name, **kwargs
                )

    return _embedding_services[cache_key]


def reset_embedding_service(cache_key: str = None):
    """Reset embedding service instance(s).

    Args:
        cache_key: Specific service to reset, or None to reset all.
    """
    global _embedding_services

    with _singleton_lock:
        if cache_key is None:
            # Reset all services
            for service in _embedding_services.values():
                service.unload()
            _embedding_services.clear()
        elif cache_key in _embedding_services:
            _embedding_services[cache_key].unload()
            del _embedding_services[cache_key]


class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB that uses our embedding service.

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
    def service(self) -> EmbeddingServiceBase:
        if self._service is None:
            self._service = get_embedding_service(self.model_name)
        return self._service

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for ChromaDB."""
        return self.service.embed_batch(input, is_query=False)


# Convenience function for quick embedding
def embed(
    texts: Union[str, List[str]], is_query: bool = False
) -> Union[List[float], List[List[float]]]:
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


# Backward compatibility
EmbeddingService = NativeEmbeddingService
