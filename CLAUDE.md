# CodeGrok MCP - Project Context for AI Assistants

This document provides comprehensive context about the CodeGrok MCP project for AI assistants working with this codebase.

## Project Overview

**Name**: CodeGrok MCP
**Version**: 0.1.0
**Type**: MCP Server for Semantic Code Search
**License**: MIT
**Python**: 3.10+

### Purpose

CodeGrok MCP is a specialized Model Context Protocol (MCP) server that enables AI assistants to perform semantic code search and understanding. It's a focused, MCP-only derivative of the CodeGrok project, removing CLI and LLM components to provide a lightweight code indexing and search service.

### What Makes This Unique

- **MCP-Only**: No CLI, no built-in LLM - purely an MCP service
- **No External Dependencies**: All processing happens locally
- **Fast Indexing**: Multi-threaded parsing and embedding generation
- **Smart Chunking**: Maintains code context across chunk boundaries
- **Incremental Updates**: Re-index only changed files

## Architecture Overview

### High-Level Flow

```
User Request (via MCP Client)
    ↓
MCP Server (server.py)
    ↓
State Management (state.py)
    ↓
Source Retriever (source_retriever.py)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│ Tree-Sitter     │ Embedding        │ ChromaDB        │
│ Parser          │ Service          │ Storage         │
└─────────────────┴──────────────────┴─────────────────┘
```

### Package Structure

```
CodeGrok_mcp/
├── pyproject.toml              # Package configuration and dependencies
├── README.md                   # User-facing documentation
├── CLAUDE.md                   # This file - AI assistant context
├── docs/
│   └── IMPLEMENTATION_PLAN.md  # Detailed implementation roadmap
└── src/codegrok_mcp/
    ├── __init__.py
    ├── core/                   # Core data structures
    │   ├── __init__.py
    │   ├── models.py          # Symbol, ParsedFile, CodebaseIndex
    │   └── interfaces.py      # IParser interface
    ├── parsers/               # Code parsing layer
    │   ├── __init__.py
    │   ├── treesitter_parser.py    # Tree-sitter implementation
    │   └── language_configs.py     # Language-specific AST mappings
    ├── indexing/              # Indexing and search layer
    │   ├── __init__.py
    │   ├── embedding_service.py    # Sentence-transformers wrapper
    │   ├── parallel_indexer.py     # Multi-threaded file processing
    │   └── source_retriever.py     # Main orchestration (no LLM)
    └── mcp/                   # MCP server layer
        ├── __init__.py
        ├── state.py           # Global state management
        └── server.py          # FastMCP server with 6 tools
```

## Key Files and Their Purposes

### Core Layer (`core/`)

#### `models.py`
Defines fundamental data structures:

- **`SymbolType`**: Enum for code symbols (FUNCTION, CLASS, METHOD, etc.)
- **`Symbol`**: Represents a parsed code symbol
  - Fields: `name`, `type`, `signature`, `line_start`, `line_end`, `file_path`
- **`ParsedFile`**: Container for parsed file data
  - Fields: `file_path`, `symbols`, `content`, `language`
- **`CodebaseIndex`**: Collection of parsed files
  - Methods for file management and symbol extraction

**Key Insight**: These models are language-agnostic and represent the normalized output from any parser.

#### `interfaces.py`
Defines the parser contract:

- **`IParser`**: Abstract interface for code parsers
  - `parse_file()`: Parse a single file
  - `parse_directory()`: Parse an entire directory
  - `supported_extensions()`: List supported file types

**Why This Exists**: Allows swapping parser implementations without changing dependent code.

### Parsers Layer (`parsers/`)

#### `treesitter_parser.py`
Tree-sitter implementation of IParser:

- **Multi-language support**: 9 languages via tree-sitter-languages
- **Symbol extraction**: Walks AST to find functions, classes, methods, variables
- **Robust error handling**: Continues on parse errors, logs issues
- **Language detection**: Auto-detects language from file extension

**Key Methods**:
- `parse_file(file_path)`: Returns `ParsedFile` object
- `parse_directory(dir_path, extensions)`: Returns `CodebaseIndex`
- `_extract_symbols(tree, source, language)`: AST traversal logic

**Performance**: Single-threaded, but called from parallel indexer.

#### `language_configs.py`
AST node type mappings for each language:

- Maps Tree-sitter node types to SymbolType
- Examples:
  - Python: `function_definition` → `FUNCTION`
  - JavaScript: `class_declaration` → `CLASS`
  - Go: `method_declaration` → `METHOD`

**Why This Exists**: Tree-sitter uses different node names across languages. This normalizes them.

### Indexing Layer (`indexing/`)

#### `embedding_service.py`
Manages embedding generation:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Device**: Auto-detects CUDA/MPS/CPU
- **Caching**: Batch processing for efficiency

**Key Methods**:
- `embed_texts(texts)`: Generate embeddings for text list
- `embed_single(text)`: Generate single embedding

**Performance**: ~200ms for batch of 50 texts on CPU, faster on GPU.

#### `parallel_indexer.py`
Thread-pool based file processing:

- **Default**: 8 worker threads
- **Chunking**: Splits large files into overlapping segments
- **Symbol-aware**: Aligns chunks with symbol boundaries when possible

**Key Methods**:
- `index_files(file_paths, chunk_size, chunk_overlap)`: Parallel processing
- `_process_file(file_path, chunk_size, chunk_overlap)`: Single file handler

**Output**: List of chunks with metadata (file, line range, symbols, embeddings)

#### `source_retriever.py`
Main orchestration class (refactored from rag_chat.py):

**Critical Refactoring**: Removed all LLM-related code (ask, ask_stream, reranking)

**Core Responsibilities**:
1. **Indexing**: Coordinates parsing, chunking, embedding, storage
2. **Search**: Semantic similarity search via ChromaDB
3. **Persistence**: Saves/loads metadata and embeddings
4. **Incremental Updates**: Detects file changes via SHA256 hashing

**Key Methods**:
- `index_codebase()`: Full indexing pipeline
- `get_sources_for_question(query, top_k)`: Semantic search
- `incremental_reindex()`: Re-index only changed files
- `load_existing_index()`: Load from `.codegrok/` directory
- `get_stats()`: Return indexing statistics

**Storage Structure**:
```
project-root/.codegrok/
├── chromadb/              # ChromaDB vector store
│   └── [binary files]
└── metadata.json          # File hashes, timestamps, config
```

**Metadata Schema**:
```json
{
  "project_name": "my-app",
  "project_path": "/path/to/project",
  "file_hashes": {
    "src/main.py": "abc123...",
    "src/utils.py": "def456..."
  },
  "chunk_size": 100,
  "chunk_overlap": 20,
  "indexed_at": "2025-12-12T10:30:00"
}
```

### MCP Layer (`mcp/`)

#### `state.py`
Global state management:

- **Purpose**: Maintains single SourceRetriever instance across MCP calls
- **Pattern**: Singleton-like global state
- **Thread Safety**: Not thread-safe (MCP is single-threaded)

**Functions**:
- `get_chat()`: Returns current SourceRetriever or None
- `set_chat(chat)`: Sets global SourceRetriever instance
- `clear_chat()`: Clears global state

**Why This Exists**: MCP tools are stateless functions. This provides persistence between calls.

#### `server.py`
FastMCP server implementation:

**Entry Point**: `main()` function (called by `codegrok-mcp` command)

**MCP Tools** (6 total):

1. **`learn`**: Index a codebase
   - Creates SourceRetriever, calls `index_codebase()`, stores in state

2. **`relearn`**: Incremental re-indexing
   - Requires existing index, calls `incremental_reindex()`

3. **`load`**: Load existing index
   - Creates SourceRetriever from `.codegrok/` directory

4. **`get_sources`**: Semantic search
   - Calls `get_sources_for_question(query, top_k)`
   - Returns list of code chunks with file paths

5. **`get_stats`**: Index statistics
   - Returns files count, symbols count, chunks count

6. **`list_supported_languages`**: List supported languages
   - Returns static mapping of languages to extensions

**Transport**: stdio (reads from stdin, writes to stdout)

**Server Instructions**: Embedded prompt explaining tool purposes to MCP clients

## Dependencies Explained

### Production Dependencies

| Package | Purpose | Why It's Critical |
|---------|---------|-------------------|
| `tree-sitter` | Parse source code | AST parsing for symbol extraction |
| `tree-sitter-languages` | Language grammars | Supports 9 languages out of the box |
| `chromadb` | Vector database | Stores embeddings, performs similarity search |
| `sentence-transformers` | Embeddings | Converts code to semantic vectors |
| `torch` | ML framework | Required by sentence-transformers |
| `fastmcp` | MCP server | Handles MCP protocol and tool registration |
| `numpy` | Arrays | Data manipulation for embeddings |
| `rich` | Terminal output | Formatted logging (minimal use) |
| `einops` | Tensor ops | Dependency of sentence-transformers |

### Development Dependencies

- `pytest`: Testing framework
- `black`: Code formatter (line length 100)
- `mypy`: Type checker
- `flake8`: Linter

### Notable Exclusions

**Removed from parent CodeGrok project**:
- `llama-cpp-python`: LLM inference (no LLM in this version)
- `ollama`: Alternative LLM backend (removed)
- `click`: CLI framework (no CLI)
- `questionary`: Interactive prompts (no CLI)

## Data Flow

### Indexing Flow

```
1. User calls `learn` tool
   ↓
2. MCP server receives request
   ↓
3. Creates SourceRetriever instance
   ↓
4. SourceRetriever.index_codebase()
   ↓
5. Discover files (by extension)
   ↓
6. ParallelIndexer.index_files()
   ├─ Thread 1: Parse file A → Extract symbols → Chunk → Embed
   ├─ Thread 2: Parse file B → Extract symbols → Chunk → Embed
   └─ Thread N: Parse file N → Extract symbols → Chunk → Embed
   ↓
7. Store in ChromaDB
   ↓
8. Save metadata.json (file hashes)
   ↓
9. Store SourceRetriever in state
   ↓
10. Return statistics to user
```

### Search Flow

```
1. User calls `get_sources` tool with query
   ↓
2. MCP server retrieves SourceRetriever from state
   ↓
3. SourceRetriever.get_sources_for_question(query, top_k)
   ↓
4. EmbeddingService.embed_single(query)
   ↓
5. ChromaDB.query(embedding, n_results=top_k)
   ↓
6. Format results (file paths, code snippets, scores)
   ↓
7. Return to user
```

### Incremental Reindex Flow

```
1. User calls `relearn` tool
   ↓
2. Load metadata.json (file hashes)
   ↓
3. Scan project directory
   ↓
4. For each file:
   ├─ Compute SHA256 hash
   ├─ Compare with stored hash
   ├─ If changed/new: Add to reindex list
   └─ If deleted: Mark for removal
   ↓
5. Re-index changed files (same as indexing flow)
   ↓
6. Remove deleted files from ChromaDB
   ↓
7. Update metadata.json
   ↓
8. Return statistics (updated, added, removed)
```

## Common Development Tasks

### Adding a New Language

1. **Verify Tree-sitter support**: Check if `tree-sitter-languages` includes it
2. **Add to language_configs.py**: Map AST node types to SymbolType
3. **Update SUPPORTED_LANGUAGES**: Add extensions in `treesitter_parser.py`
4. **Test**: Create test file and verify symbol extraction

### Modifying Chunk Strategy

Edit `parallel_indexer.py`:
- Adjust default `chunk_size` and `chunk_overlap`
- Modify `_process_file()` to change chunking logic
- Consider symbol boundaries vs line-based chunking

### Changing Embedding Model

Edit `embedding_service.py`:
- Change `model_name` in `__init__`
- Note: Requires re-indexing all projects (embedding dimensions may change)
- Test performance impact on CPU vs GPU

### Adding a New MCP Tool

1. **Edit `server.py`**: Add `@mcp.tool` decorated function
2. **Add docstring**: Explain parameters and return value (shown to MCP clients)
3. **Use state management**: Get/set SourceRetriever via `state.py`
4. **Handle errors**: Wrap in try/except, return user-friendly messages
5. **Update README.md**: Document the new tool

## Testing Strategy

### Unit Tests
- Test individual components (parser, embedder, chunker)
- Mock expensive operations (embedding generation, file I/O)

### Integration Tests
- Test full indexing pipeline on small sample projects
- Verify search returns expected results
- Test incremental reindexing logic

### MCP Testing
- Use MCP Inspector or test client
- Verify tool parameters and responses
- Test error handling

## Performance Considerations

### Indexing Performance

**Bottlenecks**:
1. Tree-sitter parsing (CPU-bound)
2. Embedding generation (GPU-accelerated if available)
3. ChromaDB insertion (I/O-bound)

**Optimizations**:
- Parallel file processing (default 8 threads)
- Batch embedding generation (50 texts at a time)
- Incremental indexing (only changed files)

**Typical Performance**:
- Small project (100 files): ~30 seconds
- Medium project (1000 files): ~5 minutes
- Large project (10000 files): ~30 minutes

### Search Performance

**Typical**: 100-500ms for semantic search
**Factors**:
- Index size (number of chunks)
- top_k parameter
- ChromaDB query optimization

### Memory Usage

**Indexing**: ~2-4GB RAM for typical projects
**Search**: ~500MB-1GB RAM (model + index in memory)

**Tips**:
- Reduce chunk_size for large codebases
- Use GPU for faster embedding generation
- Close other applications during indexing

## Common Issues and Solutions

### Issue: "No module named 'codegrok_mcp'"
**Cause**: Package not installed
**Solution**: Run `pip install -e .` from project root

### Issue: Indexing crashes with OOM
**Cause**: Too many large files being processed
**Solution**: Reduce chunk_size or number of parallel workers

### Issue: Search returns irrelevant results
**Cause**: Query too vague or index quality issues
**Solution**:
- Use more specific queries
- Try re-indexing with smaller chunk_size
- Verify correct files were indexed

### Issue: MCP client can't find server
**Cause**: Installation path issues
**Solution**:
- Verify `codegrok-mcp` is in PATH
- Use full path to command in MCP config
- Check Python environment is activated

## Extension Points

### Custom Parsers
Implement `IParser` interface for new parsing backends:
- LSP-based parser
- Regex-based simple parser
- API-based parser (GitHub CodeQL)

### Custom Storage
Replace ChromaDB with alternative vector database:
- Modify `source_retriever.py`
- Implement same query interface
- Consider migration scripts

### Custom Embeddings
Replace sentence-transformers with alternatives:
- OpenAI embeddings (requires API key)
- Cohere embeddings
- Custom fine-tuned models

## Best Practices for Working with This Codebase

1. **Preserve Type Hints**: All code uses type annotations - maintain this
2. **Follow Black Formatting**: Line length 100, run before committing
3. **Update Both READMEs**: User-facing and AI-facing documentation
4. **Test with Real Codebases**: Use diverse projects for testing
5. **Handle Errors Gracefully**: Return informative messages to MCP clients
6. **Log Important Events**: Use rich logging for debugging
7. **Version Metadata**: Track metadata schema versions for migrations

## Debugging Tips

### Enable Verbose Logging
Set environment variable before running:
```bash
export LOG_LEVEL=DEBUG
codegrok-mcp
```

### Inspect ChromaDB
```python
import chromadb
client = chromadb.PersistentClient(path="/path/to/project/.codegrok/chromadb")
collection = client.get_collection("codebase_chunks")
print(collection.count())
print(collection.peek())
```

### Test Parsing Directly
```python
from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser
parser = TreeSitterParser()
parsed = parser.parse_file("/path/to/file.py")
print(parsed.symbols)
```

### Test Embedding
```python
from codegrok_mcp.indexing.embedding_service import EmbeddingService
embedder = EmbeddingService()
embedding = embedder.embed_single("test query")
print(embedding.shape)
```

## Project Lineage

**Parent Project**: CodeGrok (full-featured CLI tool with LLM)

**Key Differences**:
- Removed: CLI interface, LLM support, chat history, re-ranking
- Kept: Parsing, indexing, embedding, MCP server
- Modified: `rag_chat.py` → `source_retriever.py` (removed LLM methods)

**Why This Fork Exists**:
- Lightweight MCP server without heavy LLM dependencies
- Focus on code search as a service, not standalone tool
- Better integration with MCP clients like Claude Desktop

## Future Enhancements (Ideas)

1. **Workspace Support**: Index multiple projects simultaneously
2. **Symbol-Specific Search**: Search only functions, classes, etc.
3. **Cross-Reference Analysis**: Find symbol usages and definitions
4. **Git Integration**: Track changes per commit
5. **Language-Specific Filters**: Search only Python, only JS, etc.
6. **Custom Metadata**: Attach tags or notes to files
7. **Export/Import**: Share indexes between machines
8. **Performance Metrics**: Track indexing time, search latency

## Quick Reference

### Commands
- **Install**: `pip install .`
- **Run Server**: `codegrok-mcp`
- **Format Code**: `black src/`
- **Run Tests**: `pytest`
- **Type Check**: `mypy src/`

### File Paths (Absolute)
- Project Root: `/home/rdondeti/Code/Desktop/CodeGrok_mcp`
- Source Code: `/home/rdondeti/Code/Desktop/CodeGrok_mcp/src/codegrok_mcp`
- MCP Server: `/home/rdondeti/Code/Desktop/CodeGrok_mcp/src/codegrok_mcp/mcp/server.py`
- Main Logic: `/home/rdondeti/Code/Desktop/CodeGrok_mcp/src/codegrok_mcp/indexing/source_retriever.py`

### Key Constants
- Default Chunk Size: 100 lines
- Default Chunk Overlap: 20 lines
- Default top_k: 5 results
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding Dimensions: 384
- Storage Directory: `.codegrok/`

## Conclusion

CodeGrok MCP is a focused, efficient MCP server for semantic code search. It trades features (LLM, CLI) for simplicity and speed. When working with this codebase, remember:

- It's MCP-only - all interaction happens through MCP tools
- It's local-first - no external API calls
- It's incremental - only re-index what changed
- It's extensible - clean interfaces for customization

For questions about implementation details, refer to the source code with the context provided in this document.
