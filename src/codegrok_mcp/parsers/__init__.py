"""
Parsers for CodeGrok MCP.

This module provides code parsing functionality using tree-sitter
for multi-language support.
"""

from .treesitter_parser import TreeSitterParser, ThreadLocalParserFactory
from .language_configs import (
    get_language_for_file,
    get_config_for_language,
    get_supported_extensions,
    EXTENSION_MAP,
    LANGUAGE_CONFIGS,
)

__all__ = [
    "TreeSitterParser",
    "ThreadLocalParserFactory",
    "get_language_for_file",
    "get_config_for_language",
    "get_supported_extensions",
    "EXTENSION_MAP",
    "LANGUAGE_CONFIGS",
]
