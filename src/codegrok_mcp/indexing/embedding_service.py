"""
Embedding service using Ollama API only.

No local AI/CUDA/torch - all embeddings are generated via Ollama.
Requires Ollama running locally (or CODEGROK_OLLAMA_URL) with an embedding model.

Configuration via environment variables:
    CODEGROK_OLLAMA_URL=http://localhost:11434  (default)
    CODEGROK_OLLAMA_MODEL=nomic-embed-text  (default)
    CODEGROK_OLLAMA_DIMENSIONS=768  (default for nomic-embed-text; required if using other model)

Usage:
    from codegrok_mcp.indexing.embedding_service import get_embedding_service

    service = get_embedding_service()
    embedding = service.embed("def hello(): pass")
    embeddings = service.embed_batch(["text1", "text2", "text3"])
"""

import os
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Union

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_DIMENSIONS = 768


class EmbeddingServiceBase(ABC):
    """
    Abstract base class for embedding services.
    Single implementation: OllamaEmbeddingService (Ollama API).
    """

    def __init__(self, dimensions: int, batch_size: int = 32, show_progress: bool = False):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._lock = threading.Lock()

        self.stats = {
            "total_embeddings": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self._embed_cached = lru_cache(maxsize=1000)(self._embed_single_uncached)

    @abstractmethod
    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """Embed a single text without caching. Returns tuple for hashability in LRU cache."""
        pass

    @abstractmethod
    def embed_batch(
        self, texts: List[str], is_query: bool = False, batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Embed a batch of texts."""
        pass

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Embed a single text with LRU caching."""
        cache_info_before = self._embed_cached.cache_info()
        embedding_tuple = self._embed_cached(text, is_query)
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
        stats["embeddings_per_second"] = (
            stats["total_embeddings"] / stats["total_time"] if stats["total_time"] > 0 else 0
        )
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
        """Unload / free resources."""
        pass


class OllamaEmbeddingService(EmbeddingServiceBase):
    """
    Embedding service using Ollama API.

    - Remote embedding generation via Ollama (no local CUDA/torch)
    - Compatible with any Ollama embedding model (e.g. nomic-embed-text)
    - Batch processing with configurable size and retry logic
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        dimensions: int = DEFAULT_OLLAMA_DIMENSIONS,
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: int = 60,
        show_progress: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests is required for Ollama embedding. Install with: pip install requests"
            )

        super().__init__(dimensions=dimensions, batch_size=batch_size, show_progress=show_progress)

    def _embed_single_uncached(self, text: str, is_query: bool) -> tuple:
        """Embed a single text via Ollama API."""
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
                time.sleep(0.5 * (attempt + 1))

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
            batch_embeddings = self._embed_batch_parallel(batch)
            embeddings.extend(batch_embeddings)

            if self.show_progress and i % (batch_size * 10) == 0:
                progress = (i + len(batch)) / len(texts) * 100
                print(f"Embedding progress: {progress:.1f}%", end="\r")

        elapsed = time.time() - start_time
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
        """Clear cache; model lives in Ollama server."""
        self._embed_cached.cache_clear()


def create_embedding_service(**kwargs) -> EmbeddingServiceBase:
    """
    Create Ollama embedding service from environment configuration.

    Env: CODEGROK_OLLAMA_URL, CODEGROK_OLLAMA_MODEL, CODEGROK_OLLAMA_DIMENSIONS
    """
    base_url = os.environ.get("CODEGROK_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("CODEGROK_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    dimensions_str = os.environ.get("CODEGROK_OLLAMA_DIMENSIONS")
    dimensions = int(dimensions_str) if dimensions_str else DEFAULT_OLLAMA_DIMENSIONS

    return OllamaEmbeddingService(
        base_url=base_url,
        model=model,
        dimensions=dimensions,
        **{k: v for k, v in kwargs.items() if k in ("batch_size", "max_retries", "timeout", "show_progress")},
    )


_embedding_services: Dict[str, EmbeddingServiceBase] = {}
_singleton_lock = threading.Lock()


def get_embedding_service(**kwargs) -> EmbeddingServiceBase:
    """
    Get singleton Ollama embedding service (cache keyed by model).
    """
    global _embedding_services
    model = os.environ.get("CODEGROK_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    cache_key = f"ollama:{model}"

    if cache_key not in _embedding_services:
        with _singleton_lock:
            if cache_key not in _embedding_services:
                _embedding_services[cache_key] = create_embedding_service(**kwargs)

    return _embedding_services[cache_key]


def reset_embedding_service(cache_key: Optional[str] = None):
    """Reset embedding service instance(s)."""
    global _embedding_services
    with _singleton_lock:
        if cache_key is None:
            for service in _embedding_services.values():
                service.unload()
            _embedding_services.clear()
        elif cache_key in _embedding_services:
            _embedding_services[cache_key].unload()
            del _embedding_services[cache_key]


class ChromaDBEmbeddingFunction:
    """ChromaDB embedding function using Ollama."""

    def __init__(self):
        self._service = None

    @property
    def service(self) -> EmbeddingServiceBase:
        if self._service is None:
            self._service = get_embedding_service()
        return self._service

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.service.embed_batch(input, is_query=False)


def embed(
    texts: Union[str, List[str]], is_query: bool = False
) -> Union[List[float], List[List[float]]]:
    """Quick embed: single text or list of texts."""
    service = get_embedding_service()
    if isinstance(texts, str):
        return service.embed(texts, is_query=is_query)
    return service.embed_batch(texts, is_query=is_query)
