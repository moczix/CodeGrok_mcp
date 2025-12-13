"""Session state management for MCP server."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from codegrok_mcp.indexing.source_retriever import SourceRetriever


@dataclass
class MCPSessionState:
    """Singleton state for MCP server session."""
    retriever: Optional["SourceRetriever"] = None
    codebase_path: Optional[Path] = None
    @property
    def is_loaded(self) -> bool:
        """Check if a codebase is currently loaded."""
        return self.retriever is not None


_state: Optional[MCPSessionState] = None


def get_state() -> MCPSessionState:
    """Get or create the singleton state instance."""
    global _state
    if _state is None:
        _state = MCPSessionState()
    return _state


def reset_state() -> None:
    """Reset the singleton state (useful for testing)."""
    global _state
    _state = None
