"""
MCP Server for CodeGrok - Semantic code search and source retrieval.

This module provides an MCP server that exposes CodeGrok's semantic search
functionality through the Model Context Protocol using stdio transport.

Usage:
    codegrok-mcp  # Run as stdio MCP server

Tools:
    - learn: Index a codebase (smart mode: auto-detects existing index)
        - mode='auto': Incremental reindex if exists, full index if new
        - mode='full': Force complete re-index
        - mode='load_only': Just load existing index
    - get_sources: Semantic search for code
    - get_stats: Get indexing statistics
    - list_supported_languages: List supported file extensions
"""

from typing import Optional, List, Dict, Any, Annotated, Callable
from pathlib import Path
import asyncio

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from codegrok_mcp.mcp.state import get_state
# Lazy import SourceRetriever to avoid heavy startup cost
# from codegrok_mcp.indexing.source_retriever import SourceRetriever, SUPPORTED_EXTENSIONS
from codegrok_mcp.parsers.language_configs import EXTENSION_MAP

# Derived supported extensions list (lightweight)
SUPPORTED_EXTENSIONS = sorted(list(EXTENSION_MAP.keys()))


# Storage constants (same as CLI)
CODEGROK_DIR = '.codegrok'
CHROMA_DIR = 'chroma'
METADATA_FILE = 'metadata.json'

# Initialize FastMCP server
mcp = FastMCP(
    name="CodeGrok",
    instructions="Semantic code search and source retrieval for codebase understanding"
)


def _get_codegrok_paths(codebase_path: Path) -> Dict[str, Path]:
    """Get paths to .codegrok storage locations."""
    codegrok_dir = codebase_path / CODEGROK_DIR
    return {
        'codegrok_dir': codegrok_dir,
        'chroma_path': codegrok_dir / CHROMA_DIR,
        'metadata_path': codegrok_dir / METADATA_FILE,
    }


def _has_valid_index(paths: Dict[str, Path]) -> bool:
    """Check if a valid CodeGrok index exists at the given paths."""
    return (
        paths['codegrok_dir'].exists() and
        paths['metadata_path'].exists() and
        paths['chroma_path'].exists()
    )


def _create_learn_progress_callback(ctx: Context, loop) -> Callable:
    """Create a progress callback that reports indexing progress to MCP client."""
    def callback(event_type: str, data: dict):
        progress = 0
        message = ""

        if event_type == "files_found":
            progress = 5
            message = f"Found {len(data['files'])} files..."
        elif event_type == "parsing_start":
            progress = 10
            message = f"Parsing {data['total']} files..."
        elif event_type == "chunks_created":
            progress = 30
            message = f"Created {data['total']} code chunks..."
        elif event_type == "embedding_start":
            progress = 35
            message = f"Generating embeddings for {data['total']} chunks..."
        elif event_type == "embedding_progress":
            # Scale embedding progress (35-95%)
            pct = data['current'] / data['total'] if data['total'] > 0 else 1
            progress = 35 + int(pct * 60)
            message = f"Embedding... ({data['current']}/{data['total']} chunks)"
        elif event_type == "complete":
            progress = 100
            message = "Indexing complete!"

        if progress > 0:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, 100, message),
                loop
            )
    return callback


def _create_relearn_progress_callback(ctx: Context, loop) -> Callable:
    """Create a progress callback that reports reindexing progress to MCP client."""
    def callback(event_type: str, data: dict):
        progress = 0
        message = ""

        if event_type == "changes_detected":
            progress = 10
            message = f"Found {data['new']} new, {data['modified']} modified files..."
        elif event_type == "parsing_start":
            progress = 20
            message = f"Parsing {data['total']} changed files..."
        elif event_type == "embedding_start":
            progress = 40
            message = f"Updating embeddings for {data['total']} chunks..."
        elif event_type == "complete":
            progress = 100
            message = "Re-indexing complete!"

        if progress > 0:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, 100, message),
                loop
            )
    return callback


@mcp.tool(
    name="learn",
    description="""Index a codebase for semantic search.

Modes:
- auto (default): Smart detection. If index exists, updates incrementally. If new, does full index.
- full: Force complete re-index (destroys existing index).
- load_only: Just load existing index without any indexing.

Creates a .codegrok/ folder in the codebase directory."""
)
async def learn(
    path: Annotated[str, Field(description="Absolute path to the codebase directory to index")],
    mode: Annotated[
        str,
        Field(description="Indexing mode: 'auto' (smart detection), 'full' (force re-index), 'load_only' (just load)")
    ] = "auto",
    file_extensions: Annotated[
        Optional[List[str]],
        Field(description="Optional list of file extensions to include (e.g., ['.py', '.js']). Defaults to all supported extensions.")
    ] = None,
    embedding_model: Annotated[
        str,
        Field(description="Embedding model to use (default: nomic-embed-code)")
    ] = "nomic-embed-code",
    ctx: Context = None
) -> Dict[str, Any]:
    """Index a codebase with smart mode detection."""
    state = get_state()

    # Validate mode
    valid_modes = ("auto", "full", "load_only")
    if mode not in valid_modes:
        raise ToolError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    # Validate path
    codebase_path = Path(path).resolve()
    if not codebase_path.exists():
        raise ToolError(f"Path does not exist: {codebase_path}")
    if not codebase_path.is_dir():
        raise ToolError(f"Path is not a directory: {codebase_path}")

    # Get storage paths
    paths = _get_codegrok_paths(codebase_path)
    has_existing = _has_valid_index(paths)

    # Handle load_only mode
    if mode == "load_only":
        if not has_existing:
            raise ToolError(
                f"No existing index found at {codebase_path}. "
                "Use mode='auto' or mode='full' to create one."
            )
        return await _load_existing_index(codebase_path, paths, state, embedding_model)

    # Handle auto mode with existing index -> incremental reindex
    if mode == "auto" and has_existing:
        return await _incremental_reindex(codebase_path, paths, state, embedding_model, ctx)

    # Full index: mode == "full" OR (mode == "auto" and no existing index)
    return await _full_index(codebase_path, paths, state, file_extensions, embedding_model, ctx)


async def _load_existing_index(
    codebase_path: Path,
    paths: Dict[str, Path],
    state,
    embedding_model: str
) -> Dict[str, Any]:
    """Load an existing index without any reindexing."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    retriever = SourceRetriever(
        codebase_path=str(codebase_path),
        embedding_model=embedding_model,
        verbose=False,
        persist_path=str(paths['chroma_path'])
    )

    if not retriever.load_existing_index():
        raise ToolError(f"Failed to load index from {paths['chroma_path']}")

    metadata = retriever.load_metadata(str(paths['metadata_path']))
    stats = metadata.get('stats', {}) if metadata else {}
    indexed_at = metadata.get('indexed_at') if metadata else None

    state.retriever = retriever
    state.codebase_path = codebase_path

    return {
        "success": True,
        "mode_used": "load_only",
        "message": f"Loaded existing index for {codebase_path.name}",
        "stats": stats,
        "indexed_at": indexed_at
    }


async def _incremental_reindex(
    codebase_path: Path,
    paths: Dict[str, Path],
    state,
    embedding_model: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Load existing index and perform incremental reindex."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    retriever = SourceRetriever(
        codebase_path=str(codebase_path),
        embedding_model=embedding_model,
        verbose=False,
        persist_path=str(paths['chroma_path'])
    )

    if not retriever.load_existing_index():
        raise ToolError(f"Failed to load existing index from {paths['chroma_path']}")

    # Load metadata to get file mtimes for incremental detection
    retriever.load_metadata(str(paths['metadata_path']))

    # Create progress callback if context available
    progress_callback = None
    if ctx:
        loop = asyncio.get_event_loop()
        progress_callback = _create_relearn_progress_callback(ctx, loop)

    # Perform incremental reindex
    result = retriever.incremental_reindex(progress_callback=progress_callback)

    # Save updated metadata
    retriever.save_metadata(str(paths['metadata_path']))

    state.retriever = retriever
    state.codebase_path = codebase_path

    return {
        "success": True,
        "mode_used": "incremental",
        "message": f"Incremental reindex complete for {codebase_path.name}",
        **result
    }


async def _full_index(
    codebase_path: Path,
    paths: Dict[str, Path],
    state,
    file_extensions: Optional[List[str]],
    embedding_model: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """Perform full index (creates or replaces existing index)."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    # Create .codegrok directory
    paths['codegrok_dir'].mkdir(parents=True, exist_ok=True)

    # Create progress callback if context available
    progress_callback = None
    if ctx:
        loop = asyncio.get_event_loop()
        progress_callback = _create_learn_progress_callback(ctx, loop)

    retriever = SourceRetriever(
        codebase_path=str(codebase_path),
        embedding_model=embedding_model,
        verbose=False,
        persist_path=str(paths['chroma_path'])
    )

    # Index the codebase with progress reporting
    extensions = file_extensions if file_extensions else SUPPORTED_EXTENSIONS
    retriever.index_codebase(file_extensions=extensions, progress_callback=progress_callback)

    # Report saving phase
    if ctx:
        await ctx.report_progress(95, 100, "Saving index...")

    # Save metadata
    retriever.save_metadata(str(paths['metadata_path']))

    # Update state
    state.retriever = retriever
    state.codebase_path = codebase_path

    return {
        "success": True,
        "mode_used": "full",
        "message": f"Successfully indexed {codebase_path.name}",
        "stats": retriever.get_stats()
    }


@mcp.tool(
    name="get_sources",
    description="Get source code references relevant to a query using semantic search. Supports optional filtering by language and symbol type."
)
def get_sources(
    question: Annotated[str, Field(description="Natural language question or search query")],
    n_results: Annotated[
        int,
        Field(description="Number of source references to return (default: 10)", ge=1, le=50)
    ] = 10,
    language: Annotated[
        Optional[str],
        Field(description="Filter by language (e.g., 'python', 'javascript', 'typescript')")
    ] = None,
    symbol_type: Annotated[
        Optional[str],
        Field(description="Filter by symbol type (e.g., 'function', 'class', 'method')")
    ] = None
) -> Dict[str, Any]:
    """Get source references for a question with optional filters."""
    state = get_state()

    if not state.is_loaded:
        raise ToolError("No codebase loaded. Use 'learn' first.")

    try:
        # get_sources_for_question returns tuple: (doc_results, formatted_sources)
        doc_results, formatted_sources = state.retriever.get_sources_for_question(
            question,
            n_results=n_results,
            language=language,
            symbol_type=symbol_type
        )

        return {
            "documents": doc_results,      # Full document data with metadata
            "sources": formatted_sources   # Formatted source references for display
        }

    except Exception as e:
        raise ToolError(f"Failed to get sources: {str(e)}")


@mcp.tool(
    name="get_stats",
    description="Get statistics about the currently loaded codebase index."
)
def get_stats() -> Dict[str, Any]:
    """Get indexing statistics."""
    state = get_state()

    if not state.is_loaded:
        return {
            "loaded": False,
            "codebase_path": None,
            "stats": None
        }

    return {
        "loaded": True,
        "codebase_path": str(state.codebase_path),
        "stats": state.retriever.get_stats()
    }


@mcp.tool(
    name="list_supported_languages",
    description="List all programming languages and file extensions supported by CodeGrok."
)
def list_supported_languages() -> Dict[str, Any]:
    """List supported file extensions and languages."""
    # Group extensions by language
    languages: Dict[str, List[str]] = {}
    for ext, lang in EXTENSION_MAP.items():
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(ext)

    # Sort extensions within each language
    for lang in languages:
        languages[lang] = sorted(languages[lang])

    return {
        "extensions": sorted(EXTENSION_MAP.keys()),
        "languages": languages
    }


def main():  # pragma: no cover
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
