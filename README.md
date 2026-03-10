<div align="center">

# CodeGrok MCP

**Semantic Code Search for AI Assistants**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

*Give your AI assistant the power to truly understand your codebase*

[Features](#features) • [Quick Start](#quick-start) • [Capabilities](#-what-codegrok-can-do) • [Limitations](#-what-codegrok-cannot-do) • [Integrations](#-ai-tool-integrations) • [Use Cases](#-use-cases)

</div>

---

## What is CodeGrok MCP?

CodeGrok MCP is a **Model Context Protocol (MCP) server** that enables AI assistants to intelligently search and understand codebases using **semantic embeddings** and **Tree-sitter parsing**.

Unlike simple text search, CodeGrok understands code structure - it knows what functions, classes, and methods are, and can find relevant code even when you describe it in natural language.

```
You: "Where is authentication handled?"
CodeGrok: Returns auth middleware, login handlers, JWT validation code...
```

### Why Use CodeGrok?

**The Problem:** AI assistants have limited context windows. Sending your entire codebase is expensive and often impossible.

**The Solution:** CodeGrok indexes your code once, then AI can query semantically and receive only the 5-10 most relevant code snippets—**10-100x token reduction** vs naive "read all files" approaches.

---

## Features

- **Semantic Code Search** - Find code by meaning, not just keywords
- **9 Languages Supported** - Python, JavaScript, TypeScript, C, C++, Go, Java, Kotlin, Bash
- **28 File Extensions** - Comprehensive coverage including `.jsx`, `.tsx`, `.mjs`, `.hpp`, etc.
- **Fast Parallel Indexing** - 3-5x faster with multi-threaded parsing
- **Incremental Updates** - Only re-index changed files (auto mode)
- **Local & Private** - All data stays on your machine in `.codegrok/` folder
- **Ollama-only Embeddings** - No local AI stack (no CUDA, PyTorch, or large model downloads)
- **Lightweight Install** - Small dependency footprint; embeddings via Ollama API
- **Works with Any MCP Client** - Claude, Cursor, Cline, and more

---

## ✅ What CodeGrok CAN Do

### For Live Coding (AI-Assisted Development)

| Capability | Description |
|------------|-------------|
| **Semantic Code Search** | Natural language queries → vector similarity search against indexed code |
| **Find Code by Purpose** | Query "How does auth work?" → Returns relevant auth files with line numbers |
| **Symbol Extraction** | Extracts functions, classes, methods with signatures, docstrings, calls, imports |
| **Incremental Updates** | `learn` with auto mode only re-indexes modified files (uses file modification time) |
| **Persistent Storage** | Index survives restarts in `.codegrok/` folder |
| **Load Existing Index** | `learn` with `mode='load_only'` instantly loads previously indexed codebase |

### For Learning a New Codebase

| Capability | Description |
|------------|-------------|
| **Entry Point Discovery** | Query "main entry point" to find where execution starts |
| **Architecture Understanding** | Query "database connection" to find DB layer |
| **Domain Concepts** | Query "user authentication flow" to find auth logic |
| **Index Statistics** | See files parsed, symbols extracted, timing info |

---

## ❌ What CodeGrok CANNOT Do

> **Important:** Understanding limitations helps you use the tool effectively.

### Not Designed For

| Limitation | Explanation |
|------------|-------------|
| **Code Execution** | Pure indexing/search - no interpreter, no running tests |
| **Code Modification** | Read-only search - doesn't write or edit files |
| **Real-time File Watching** | No daemon mode - manually call `learn` again to update index |
| **Cross-repository Search** | Single codebase per index - can't search multiple projects simultaneously |
| **Find All Usages** | Finds definitions, not references (no "who calls this function?") |
| **Type Inference / LSP** | No language server - no jump-to-definition, no autocomplete |
| **Git History Analysis** | Indexes current state only - no commit history or blame |
| **Regex/Exact Search** | Semantic only - use `grep` or `ripgrep` for exact string matching |
| **Code Metrics** | No complexity scoring, no linting, no coverage data |

### Technical Constraints

| Constraint | Impact |
|------------|--------|
| **Ollama required** | Must have [Ollama](https://ollama.com) running with an embedding model (e.g. `ollama pull nomic-embed-text`) |
| **First index** | Speed depends on Ollama; typically ~20-50 chunks/min |
| **Query latency** | ~100-300ms per search (Ollama API round-trip) |

---

## Quick Start

### Prerequisites

- **Ollama** – [Install Ollama](https://ollama.com) and pull an embedding model:
  ```bash
  ollama pull nomic-embed-text
  ```

### Installation

```bash
# Clone the repository
git clone https://github.com/dondetir/CodeGrok_mcp
cd CodeGrok_mcp

# Option 1: Use setup script (recommended)
./setup.sh              # Linux/macOS
# or
.\setup.ps1             # Windows PowerShell

# Option 2: Manual install (lightweight – no CUDA/PyTorch)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e .

# Verify installation
codegrok-mcp --help
```

Optional environment variables (defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGROK_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `CODEGROK_OLLAMA_MODEL` | `nomic-embed-text` | Embedding model name |
| `CODEGROK_OLLAMA_DIMENSIONS` | `768` | Vector size (must match model) |

**Setup script options:**
| Flag | Description |
|------|-------------|
| `--clean` | Remove existing venv before creating new |
| `--prod` | Install production dependencies only |
| `--no-verify` | Skip verification step |

### First Index

Once integrated with your AI tool (see below), ask your assistant:

```
"Learn my codebase at /path/to/my/project"
```

Then search:

```
"Find how API endpoints are defined"
"Where is error handling implemented?"
"Show me the database models"
```

---

## 🎯 Use Cases

### Use Case 1: Live Coding with AI

**How CodeGrok Saves Tokens:**

```
Without CodeGrok:
  AI tries to read entire codebase → exceeds context window → fails or costs $$

With CodeGrok:
  AI: "I need to add a new route"
    ↓ calls get_sources("Express route definition")
  CodeGrok: Returns routes/api.js:15, routes/auth.js:8
    ↓ AI reads only those 2 files
  Result: 10-100x fewer tokens, faster responses
```

### Use Case 2: Learning a New Codebase

```
Step 1: "Learn my codebase at ~/projects/big-app"
Step 2: "Where is the main entry point?"
Step 3: "How is authentication implemented?"
Step 4: "Find the database connection logic"
Step 5: "Show me how API errors are handled"
```

### Use Case 3: Code Review Assistance

```
"Find all functions that handle user input"
"Where is validation performed?"
"Show me error handling patterns"
```

---

## 🔌 AI Tool Integrations

### Claude Code (CLI)

The easiest way to add CodeGrok to Claude Code:

```bash
# Add the MCP server
claude mcp add codegrok-mcp -- codegrok-mcp
```

Or manually add to your settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "codegrok": {
      "command": "codegrok-mcp"
    }
  }
}
```

**Usage in Claude Code:**
```
> learn my codebase at ./my-project
> find authentication logic
> where is the main entry point?
```

---

### Claude Desktop

Add to your Claude Desktop configuration:

| Platform | Config File Location |
|----------|---------------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "codegrok": {
      "command": "codegrok-mcp",
      "args": []
    }
  }
}
```

Restart Claude Desktop after saving.

---

### Cursor

Cursor supports MCP servers through its extension system:

1. **Open Settings** → Extensions → MCP
2. **Add Server Configuration**:

```json
{
  "codegrok": {
    "command": "codegrok-mcp",
    "transport": "stdio"
  }
}
```

Or add to `.cursor/mcp.json` in your project:

```json
{
  "servers": {
    "codegrok": {
      "command": "codegrok-mcp"
    }
  }
}
```

---

### Windsurf (Codeium)

Windsurf supports MCP through Cascade:

1. Open **Cascade Settings**
2. Navigate to **MCP Servers**
3. Add configuration:

```json
{
  "codegrok": {
    "command": "codegrok-mcp",
    "transport": "stdio"
  }
}
```

---

### Cline (VS Code)

Add to Cline's MCP settings in VS Code:

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Search "Cline: Open MCP Settings"
3. Add:

```json
{
  "mcpServers": {
    "codegrok": {
      "command": "codegrok-mcp"
    }
  }
}
```

---

### Zed Editor

Zed supports MCP through its assistant panel. Add to settings:

```json
{
  "assistant": {
    "mcp_servers": {
      "codegrok": {
        "command": "codegrok-mcp"
      }
    }
  }
}
```

---

### Continue (VS Code/JetBrains)

Add to your Continue configuration (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "codegrok",
      "command": "codegrok-mcp"
    }
  ]
}
```

---

### Generic MCP Client

For any MCP-compatible client, use stdio transport:

```bash
# Command to run
codegrok-mcp

# Transport
stdio (stdin/stdout)

# Protocol
Model Context Protocol (MCP)
```

---

## MCP Tools Reference

CodeGrok provides **4 tools** for AI assistants:

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `learn` | Index a codebase (smart modes) | `path` (required), `mode` (auto/full/load_only), `file_extensions`, `embedding_model` (Ollama model name) |
| `get_sources` | Semantic code search | `question` (required), `n_results` (1-50, default: 10), `language`, `symbol_type` |
| `get_stats` | Get index statistics | None |
| `list_supported_languages` | List supported languages | None |

**Learn modes:**
- `auto` (default): Smart detection - incremental reindex if exists, full index if new
- `full`: Force complete re-index (destroys existing index)
- `load_only`: Just load existing index without any indexing

### Tool Examples

#### Learn a Codebase
```json
{
  "tool": "learn",
  "arguments": {
    "path": "/home/user/my-project",
    "mode": "auto"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Indexed 150 files with 1,247 symbols",
  "stats": {
    "total_files": 150,
    "total_symbols": 1247,
    "total_chunks": 2834,
    "indexing_time": 12.5
  }
}
```

#### Search for Code
```json
{
  "tool": "get_sources",
  "arguments": {
    "question": "How is user authentication implemented?",
    "n_results": 5
  }
}
```

**Response:**
```json
{
  "sources": [
    {
      "file": "src/auth/middleware.py",
      "symbol": "authenticate_request",
      "type": "function",
      "line": 45,
      "content": "def authenticate_request(request):\n    ...",
      "score": 0.89
    }
  ]
}
```

#### Incremental Update (using learn with auto mode)
```json
{
  "tool": "learn",
  "arguments": {
    "path": "/home/user/my-project",
    "mode": "auto"
  }
}
```

**Response (when index exists):**
```json
{
  "success": true,
  "mode_used": "incremental",
  "files_added": 2,
  "files_modified": 5,
  "files_deleted": 1
}
```

---

## Supported Languages

| Language | Extensions | Parser |
|----------|------------|--------|
| **Python** | `.py`, `.pyi`, `.pyw` | tree-sitter-python |
| **JavaScript** | `.js`, `.jsx`, `.mjs`, `.cjs` | tree-sitter-javascript |
| **TypeScript** | `.ts`, `.tsx`, `.mts`, `.cts` | tree-sitter-typescript |
| **C** | `.c`, `.h` | tree-sitter-c |
| **C++** | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx` | tree-sitter-cpp |
| **Go** | `.go` | tree-sitter-go |
| **Java** | `.java` | tree-sitter-java |
| **Kotlin** | `.kt`, `.kts` | tree-sitter-kotlin |
| **Bash** | `.sh`, `.bash`, `.zsh` | tree-sitter-bash |

**Total: 9 languages, 28 file extensions**

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client                               │
│        (Claude, Cursor, Cline, etc.)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol (stdio)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   CodeGrok MCP Server                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Parsers   │  │  Embeddings │  │   Vector Storage    │  │
│  │ (Tree-sitter)│  │  (Ollama    │  │    (ChromaDB)       │  │
│  │             │  │   API)      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Indexing Pipeline

```
Source Files → Tree-sitter Parser → Symbol Extraction →
Code Chunks → Ollama Embeddings → ChromaDB Storage
```

1. **Parse**: Tree-sitter extracts functions, classes, methods with signatures
2. **Chunk**: Code is split into semantic chunks with context (docstrings, imports, calls)
3. **Embed**: Ollama API generates vector embeddings (no local CUDA/PyTorch)
4. **Store**: ChromaDB persists vectors locally in `.codegrok/`

### Search Pipeline

```
Query → Embedding → Vector Similarity → Ranked Results
```

1. **Embed Query**: Convert natural language to vector
2. **Search**: Find similar vectors in ChromaDB
3. **Return**: Top-k results with file paths, line numbers, and code snippets

### Storage

All data is stored locally in your project:

```
your-project/
└── .codegrok/
    ├── chroma/           # Vector database
    └── metadata.json     # Index metadata (stats, file mtimes)
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CODEGROK_OLLAMA_URL` | Ollama API base URL | `http://localhost:11434` |
| `CODEGROK_OLLAMA_MODEL` | Embedding model name in Ollama | `nomic-embed-text` |
| `CODEGROK_OLLAMA_DIMENSIONS` | Vector dimensions (must match model) | `768` |

### Ollama (required)

CodeGrok uses **only Ollama** for embeddings (no local PyTorch/CUDA). Install and run Ollama, then pull an embedding model:

```bash
# Install Ollama from https://ollama.com, then:
ollama pull nomic-embed-text

# Optional: custom URL or model
export CODEGROK_OLLAMA_URL=http://localhost:11434
export CODEGROK_OLLAMA_MODEL=nomic-embed-text
export CODEGROK_OLLAMA_DIMENSIONS=768   # default for nomic-embed-text

codegrok-mcp
```

**Requirements:**
- Running Ollama server (e.g. `ollama serve` or start Ollama app)
- An embedding model pulled (e.g. `ollama pull nomic-embed-text`)
- Dimensions in env must match the model (768 for nomic-embed-text)

### Embedding models (Ollama)

| Model | Dimensions | Notes |
|-------|------------|--------|
| `nomic-embed-text` | 768 | Default; good for code and text |

---

## Development

### Setup

```bash
# Clone
git clone https://github.com/rdondeti/CodeGrok_mcp.git
cd CodeGrok_mcp

# Run setup script
./setup.sh              # Linux/macOS (includes dev dependencies)
.\setup.ps1             # Windows PowerShell

# For clean reinstall:
./setup.sh --clean
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/codegrok_mcp --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -v          # Fast unit tests
pytest tests/integration/ -v   # Integration tests (uses real embeddings)
pytest tests/mcp/ -v           # MCP protocol simulation tests
```

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

---

## FAQ & Troubleshooting

<details>
<summary><strong>Server won't start</strong></summary>

```bash
# Check installation
pip show codegrok-mcp

# Check Python version (need 3.10+)
python --version

# Reinstall
pip install -e .
```
</details>

<details>
<summary><strong>Indexing is slow</strong></summary>

- Large codebases (>10k files) take longer on first index
- Use `learn` again after first index for incremental updates (auto mode)
- Close other heavy applications
- Consider indexing a subdirectory first
</details>

<details>
<summary><strong>Search returns irrelevant results</strong></summary>

- Be more specific in queries (e.g., "JWT token validation" instead of "auth")
- Re-index if codebase changed significantly
- Check that the code type you're searching exists
</details>

<details>
<summary><strong>Out of memory</strong></summary>

- Index smaller portions of the codebase
- Ensure Ollama has enough RAM for the embedding model
- Close other applications
</details>

<details>
<summary><strong>"No index loaded" error</strong></summary>

Use `learn` tool first:
```
"Learn my codebase at /path/to/project"
```
</details>

---

## Comparison with Other Tools

| Feature | CodeGrok MCP | grep/ripgrep | GitHub Search | Sourcegraph |
|---------|--------------|--------------|---------------|-------------|
| Semantic Search | ✅ | ❌ | Partial | ✅ |
| Local/Private | ✅ | ✅ | ❌ | ❌ |
| MCP Support | ✅ | ❌ | ❌ | ❌ |
| No API Keys | ✅ | ✅ | ❌ | ❌ |
| Multi-language | ✅ | ✅ | ✅ | ✅ |
| Code Structure Aware | ✅ | ❌ | Partial | ✅ |
| Offline | ✅ | ✅ | ❌ | ❌ |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black src/`)
6. Submit a Pull Request

### Development Guidelines

- Follow Black formatting (line length 100)
- Add type hints to all functions
- Write tests for new features
- Update documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Related Projects

- **[Model Context Protocol](https://modelcontextprotocol.io/)** - The protocol that powers this integration
- **[Tree-sitter](https://tree-sitter.github.io/tree-sitter/)** - Fast, accurate code parsing
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[Ollama](https://ollama.com)** - Local embedding API used by CodeGrok

---

## Support

- **Issues**: [GitHub Issues](https://github.com/rdondeti/CodeGrok_mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rdondeti/CodeGrok_mcp/discussions)

---

<div align="center">

**Made with ❤️ for developers who want AI that truly understands their code**

[⬆ Back to Top](#codegrok-mcp)

</div>
