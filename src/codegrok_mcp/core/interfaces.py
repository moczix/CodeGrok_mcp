"""
Abstract interfaces for CodeGrok MCP components.

This module defines the contracts that enable modularity and extensibility
throughout the system. Each interface represents a swappable component.

Design Philosophy:
    - Interface Segregation: Each interface has a single, focused responsibility
    - Dependency Inversion: High-level modules depend on these abstractions
    - Open/Closed: Extend functionality by implementing interfaces, not modifying them
    - Liskov Substitution: All implementations must honor the contract

When to implement each interface:
    - IParser: When adding support for a new programming language
"""

from abc import ABC, abstractmethod
from typing import List
from .models import ParsedFile


class IParser(ABC):
    """Abstract interface for code parsers.

    Parsers are responsible for reading source files and extracting structured
    symbol information. Each parser implementation supports one or more
    programming languages.

    Responsibilities:
        - Detect if a file can be parsed based on extension or content
        - Extract symbols (functions, classes, variables, etc.) from source code
        - Handle parsing errors gracefully
        - Report supported file extensions

    When to implement:
        - Adding support for a new programming language
        - Creating specialized parsers (e.g., configuration files, documentation)
        - Implementing alternative parsing strategies (AST vs regex vs tree-sitter)

    Implementation considerations:
        - Performance: Parsing should be fast enough for large codebases
        - Error handling: Invalid syntax should not crash the parser
        - Completeness: Extract all relevant symbols, not just functions
        - Accuracy: Symbol locations and signatures must be precise

    Example implementation:
        >>> class PythonParser(IParser):
        ...     def can_parse(self, filepath: str) -> bool:
        ...         return filepath.endswith(('.py', '.pyi'))
        ...
        ...     def parse_file(self, filepath: str) -> ParsedFile:
        ...         # Use tree-sitter or AST to extract symbols
        ...         return ParsedFile(filepath, "python", symbols)
        ...
        ...     def get_supported_extensions(self) -> List[str]:
        ...         return ['.py', '.pyi']
    """

    @abstractmethod
    def can_parse(self, filepath: str) -> bool:
        """Determine if this parser can handle the given file.

        This method should be fast as it's called for every file in the codebase.
        Typically checks file extension, but can also inspect file content
        if needed (e.g., detecting shell scripts by shebang).

        Args:
            filepath: Absolute path to the file to check

        Returns:
            True if this parser can parse the file, False otherwise

        Example:
            >>> parser = PythonParser()
            >>> parser.can_parse("/project/main.py")
            True
            >>> parser.can_parse("/project/config.json")
            False
        """
        pass  # pragma: no cover

    @abstractmethod
    def parse_file(self, filepath: str) -> ParsedFile:
        """Parse a source file and extract all symbols.

        This is the core parsing operation. It should:
        1. Read the file content
        2. Parse it into a syntax tree (AST, CST, or tree-sitter tree)
        3. Walk the tree to extract all symbols
        4. Capture metadata (docstrings, signatures, locations)
        5. Handle errors gracefully and record them

        Args:
            filepath: Absolute path to the file to parse

        Returns:
            ParsedFile object containing all extracted symbols and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file can't be read
            ValueError: If the file type is not supported by this parser

        Example:
            >>> parser = PythonParser()
            >>> result = parser.parse_file("/project/calculator.py")
            >>> print(f"Found {len(result.symbols)} symbols")
            Found 12 symbols
            >>> for symbol in result.symbols:
            ...     print(f"{symbol.type.value}: {symbol.name}")
            function: add
            function: subtract
            class: Calculator
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of file extensions this parser supports.

        Extensions should include the leading dot (e.g., '.py', not 'py').
        This is used for file discovery and parser selection.

        Returns:
            List of file extensions (with leading dots)

        Example:
            >>> parser = JavaScriptParser()
            >>> parser.get_supported_extensions()
            ['.js', '.jsx', '.ts', '.tsx', '.mjs']
        """
        pass  # pragma: no cover
