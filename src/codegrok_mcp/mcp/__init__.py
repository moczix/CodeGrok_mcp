"""
MCP Server module for CodeGrok.

This module provides the Model Context Protocol server implementation
for semantic code search and source retrieval.

Exports:
    - mcp: FastMCP server instance
    - main: Entry point for running the MCP server
    - get_state: Get MCP session state
    - reset_state: Reset MCP session state (for testing)
    - MCPSessionState: Session state dataclass
"""

from codegrok_mcp.mcp.server import mcp, main
from codegrok_mcp.mcp.state import get_state, reset_state, MCPSessionState

__all__ = [
    "mcp",
    "main",
    "get_state",
    "reset_state",
    "MCPSessionState",
]
