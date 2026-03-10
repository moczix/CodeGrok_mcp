"""
CodeGrok MCP Indexing Module.

Semantic search for code using Ollama embeddings (no local AI/CUDA):
- get_embedding_service: Ollama-based embedding service
- SourceRetriever: Semantic code search with ChromaDB storage
- parallel_parse_files: Parallel file parsing for faster indexing

Usage:
    from codegrok_mcp.indexing import SourceRetriever, get_embedding_service

    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )
    retriever.index_codebase()
    documents, sources = retriever.get_sources_for_question("How does auth work?")
"""

from codegrok_mcp.indexing.embedding_service import (
    EmbeddingServiceBase,
    OllamaEmbeddingService,
    get_embedding_service,
    reset_embedding_service,
    ChromaDBEmbeddingFunction,
    embed,
    create_embedding_service,
)

from codegrok_mcp.indexing.source_retriever import (
    SourceRetriever,
    CodeChunk,
    SUPPORTED_EXTENSIONS,
    count_codebase_files,
)

from codegrok_mcp.indexing.parallel_indexer import (
    parallel_parse_files,
    ParseResult,
    ParallelProgress,
)

__all__ = [
    # Embedding service (Ollama only)
    'EmbeddingServiceBase',
    'OllamaEmbeddingService',
    'get_embedding_service',
    'reset_embedding_service',
    'ChromaDBEmbeddingFunction',
    'embed',
    'create_embedding_service',
    # Source retriever
    'SourceRetriever',
    'CodeChunk',
    'SUPPORTED_EXTENSIONS',
    'count_codebase_files',
    # Parallel indexing
    'parallel_parse_files',
    'ParseResult',
    'ParallelProgress',
]
