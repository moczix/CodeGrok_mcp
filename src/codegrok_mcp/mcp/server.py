"""
MCP Server for CodeGrok - Semantic code search and source retrieval.

This module provides an MCP server that exposes CodeGrok's semantic search
functionality through the Model Context Protocol using stdio transport.

Usage:
    codegrok-mcp  # Run as stdio MCP server

Tools:
    - learn: Index a codebase (full)
    - relearn: Re-index only changed files
    - load: Load existing index
    - get_sources: Semantic search for code
    - get_stats: Get indexing statistics
    - list_supported_languages: List supported file extensions
"""

from typing import Optional, List, Dict, Any, Annotated
from pathlib import Path

from fastmcp import FastMCP
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


@mcp.tool(
    name="learn",
    description="Index a codebase for semantic search. Creates a .codegrok/ folder in the codebase directory."
)
def learn(
    path: Annotated[str, Field(description="Absolute path to the codebase directory to index")],
    file_extensions: Annotated[
        Optional[List[str]],
        Field(description="Optional list of file extensions to include (e.g., ['.py', '.js']). Defaults to all supported extensions.")
    ] = None,
    embedding_model: Annotated[
        str,
        Field(description="Embedding model to use (default: nomic-embed-text)")
    ] = "nomic-embed-text"
) -> Dict[str, Any]:
    """Index a codebase and store in .codegrok/ folder."""
    state = get_state()

    # Validate path
    codebase_path = Path(path).resolve()
    if not codebase_path.exists():
        raise ToolError(f"Path does not exist: {codebase_path}")
    if not codebase_path.is_dir():
        raise ToolError(f"Path is not a directory: {codebase_path}")

    # Get storage paths
    paths = _get_codegrok_paths(codebase_path)

    # Create .codegrok directory
    paths['codegrok_dir'].mkdir(parents=True, exist_ok=True)

    try:
        # Initialize source retriever
        from codegrok_mcp.indexing.source_retriever import SourceRetriever
        retriever = SourceRetriever(
            codebase_path=str(codebase_path),
            embedding_model=embedding_model,
            verbose=False,
            persist_path=str(paths['chroma_path'])
        )

        # Index the codebase
        extensions = file_extensions if file_extensions else SUPPORTED_EXTENSIONS
        retriever.index_codebase(file_extensions=extensions)

        # Save metadata
        retriever.save_metadata(str(paths['metadata_path']))

        # Update state
        state.retriever = retriever
        state.codebase_path = codebase_path

        return {
            "success": True,
            "message": f"Successfully indexed {codebase_path.name}",
            "stats": retriever.get_stats()
        }

    except Exception as e:
        raise ToolError(f"Failed to index codebase: {str(e)}")


@mcp.tool(
    name="relearn",
    description="Re-index only files that have changed since the last indexing. Much faster than full indexing."
)
def relearn(
    path: Annotated[
        Optional[str],
        Field(description="Path to codebase. If omitted, uses the currently loaded codebase.")
    ] = None
) -> Dict[str, Any]:
    """Re-index only changed files."""
    state = get_state()

    # If path provided, load that codebase first
    if path:
        load_result = load(path)
        if not load_result.get("success"):
            raise ToolError(f"Failed to load codebase: {load_result.get('message')}")

    if not state.is_loaded:
        raise ToolError("No codebase loaded. Use 'learn' or 'load' first, or provide a path.")

    try:
        result = state.retriever.incremental_reindex()

        # Save updated metadata
        if state.codebase_path:
            paths = _get_codegrok_paths(state.codebase_path)
            state.retriever.save_metadata(str(paths['metadata_path']))

        return {
            "success": True,
            "message": "Incremental re-index complete",
            **result
        }

    except Exception as e:
        raise ToolError(f"Failed to re-index: {str(e)}")


@mcp.tool(
    name="load",
    description="Load an existing indexed codebase from its .codegrok/ folder."
)
def load(
    path: Annotated[str, Field(description="Absolute path to the codebase directory with existing .codegrok/ folder")]
) -> Dict[str, Any]:
    """Load an existing indexed codebase."""
    state = get_state()

    # Validate path
    codebase_path = Path(path).resolve()
    if not codebase_path.exists():
        raise ToolError(f"Path does not exist: {codebase_path}")

    paths = _get_codegrok_paths(codebase_path)

    # Check if .codegrok exists
    if not paths['codegrok_dir'].exists():
        raise ToolError(
            f"No .codegrok/ found at {codebase_path}. "
            f"Use the 'learn' tool to index this codebase first."
        )

    try:
        # Initialize source retriever
        from codegrok_mcp.indexing.source_retriever import SourceRetriever
        retriever = SourceRetriever(
            codebase_path=str(codebase_path),
            embedding_model="nomic-embed-text",
            verbose=False,
            persist_path=str(paths['chroma_path'])
        )

        # Load existing index
        if not retriever.load_existing_index():
            raise ToolError(f"Failed to load index from {paths['chroma_path']}")

        # Load metadata
        metadata = retriever.load_metadata(str(paths['metadata_path']))
        stats = metadata.get('stats', {}) if metadata else {}
        indexed_at = metadata.get('indexed_at') if metadata else None

        # Update state
        state.retriever = retriever
        state.codebase_path = codebase_path

        return {
            "success": True,
            "message": f"Loaded {codebase_path.name}",
            "stats": stats,
            "indexed_at": indexed_at
        }

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to load index: {str(e)}")


@mcp.tool(
    name="get_sources",
    description="Get source code references relevant to a query using semantic search. Useful for code navigation and understanding."
)
def get_sources(
    question: Annotated[str, Field(description="Natural language question or search query")],
    n_results: Annotated[
        int,
        Field(description="Number of source references to return (default: 10)", ge=1, le=50)
    ] = 10
) -> Dict[str, Any]:
    """Get source references for a question."""
    state = get_state()

    if not state.is_loaded:
        raise ToolError("No codebase loaded. Use 'learn' or 'load' first.")

    try:
        sources = state.retriever.get_sources_for_question(question, n_results=n_results)

        return {
            "sources": sources
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
