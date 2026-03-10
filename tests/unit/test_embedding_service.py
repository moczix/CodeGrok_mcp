"""Unit tests for Ollama-only embedding service."""

import pytest
from unittest.mock import MagicMock, patch
from codegrok_mcp.indexing.embedding_service import (
    OllamaEmbeddingService,
    get_embedding_service,
    reset_embedding_service,
    create_embedding_service,
)


class TestOllamaEmbeddingService:
    def teardown_method(self):
        reset_embedding_service()

    def test_embed_empty_batch(self):
        with patch(
            "codegrok_mcp.indexing.embedding_service.OllamaEmbeddingService._requests"
        ) as mock_req:
            service = OllamaEmbeddingService(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
                dimensions=768,
            )
            service._requests = mock_req
            result = service.embed_batch([])
            assert result == []

    def test_embed_batch_calls_ollama(self):
        with patch(
            "codegrok_mcp.indexing.embedding_service.OllamaEmbeddingService._requests"
        ) as mock_req:
            mock_req.post.return_value.json.return_value = {"embedding": [0.1] * 768}
            mock_req.post.return_value.raise_for_status = MagicMock()
            service = OllamaEmbeddingService(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
                dimensions=768,
            )
            service._requests = mock_req
            result = service.embed_batch(["text1", "text2"])
            assert len(result) == 2
            assert len(result[0]) == 768
            assert mock_req.post.call_count >= 2

    def test_cache_statistics(self):
        with patch(
            "codegrok_mcp.indexing.embedding_service.OllamaEmbeddingService._requests"
        ) as mock_req:
            mock_req.post.return_value.json.return_value = {"embedding": [0.1] * 768}
            mock_req.post.return_value.raise_for_status = MagicMock()
            service = OllamaEmbeddingService(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
                dimensions=768,
            )
            service._requests = mock_req
            service.embed("test")
            service.embed("test")  # cache hit
            stats = service.get_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1

    def test_unload_clears_cache(self):
        with patch(
            "codegrok_mcp.indexing.embedding_service.OllamaEmbeddingService._requests"
        ) as mock_req:
            service = OllamaEmbeddingService(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
                dimensions=768,
            )
            service.unload()
            # No exception; cache cleared
            assert service._embed_cached.cache_info().currsize == 0

    def test_singleton_pattern(self):
        reset_embedding_service()
        with patch.dict(
            "os.environ", {"CODEGROK_OLLAMA_MODEL": "nomic-embed-text"}, clear=False
        ):
            s1 = get_embedding_service()
            s2 = get_embedding_service()
            assert s1 is s2

    def test_create_embedding_service_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "CODEGROK_OLLAMA_URL": "http://custom:11434",
                "CODEGROK_OLLAMA_MODEL": "custom-model",
                "CODEGROK_OLLAMA_DIMENSIONS": "256",
            },
            clear=False,
        ):
            reset_embedding_service()
            service = create_embedding_service()
            assert isinstance(service, OllamaEmbeddingService)
            assert service.base_url == "http://custom:11434"
            assert service.model == "custom-model"
            assert service.dimensions == 256

    def test_create_embedding_service_falls_back_to_ollama_host(self):
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_HOST": "http://remote:11434",
                "CODEGROK_OLLAMA_MODEL": "nomic-embed-text",
            },
            clear=True,
        ):
            reset_embedding_service()
            service = create_embedding_service()
            assert isinstance(service, OllamaEmbeddingService)
            assert service.base_url == "http://remote:11434"

    def test_singleton_cache_isolated_by_base_url(self):
        reset_embedding_service()
        with patch.dict(
            "os.environ",
            {
                "CODEGROK_OLLAMA_MODEL": "nomic-embed-text",
                "CODEGROK_OLLAMA_URL": "http://host-a:11434",
            },
            clear=True,
        ):
            s1 = get_embedding_service()

        with patch.dict(
            "os.environ",
            {
                "CODEGROK_OLLAMA_MODEL": "nomic-embed-text",
                "CODEGROK_OLLAMA_URL": "http://host-b:11434",
            },
            clear=True,
        ):
            s2 = get_embedding_service()

        assert s1 is not s2
