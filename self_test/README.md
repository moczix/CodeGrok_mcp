# CodeGrok MCP Self-Test Scripts

This directory contains scripts for testing and demonstrating CodeGrok MCP functionality without requiring a full MCP client setup.

## Purpose

The self-test scripts allow you to:

- Verify that CodeGrok MCP is installed and working correctly
- Test all MCP tools programmatically
- Understand how the MCP tools behave with real data
- Debug issues without needing to configure an MCP client

These scripts directly call the underlying functions that power the MCP tools, bypassing the MCP protocol layer.

## Scripts

### `demo_mcp.py`

A comprehensive demonstration script that exercises all CodeGrok MCP tools in sequence.

**What it demonstrates:**

1. `list_supported_languages` - Shows all supported programming languages and file extensions
2. `learn` (full index) - Indexes a codebase from scratch
3. `get_stats` - Retrieves indexing statistics
4. `get_sources` - Performs semantic code search with multiple queries
5. `learn` (incremental reindex) - Modifies a file and demonstrates incremental updating
6. `get_sources` (for new code) - Searches for newly added code
7. `learn` (load_only mode) - Reloads existing index from disk
8. Final statistics check

## Running the Demo

### Prerequisites

Before running the demo, ensure CodeGrok MCP is installed:

```bash
# From the project root directory
pip install -e .
```

### Usage

**Option 1: Use a temporary sample project (recommended for first-time testing)**

```bash
python self_test/demo_mcp.py
```

This will:
- Create a temporary sample Python project with realistic code
- Run all MCP tools on this sample project
- Show colored, formatted output for each tool call
- Optionally clean up the temporary files when done

**Option 2: Test on your own project**

```bash
python self_test/demo_mcp.py /path/to/your/project
```

This will index and search your actual codebase. Note: This creates a `.codegrok/` directory in your project.

### Expected Output

The demo script produces rich, formatted output showing:

1. **Supported Languages Table** - A table listing all supported programming languages and their file extensions

2. **Indexing Progress** - JSON output showing:
   ```json
   {
     "status": "success",
     "stats": {
       "total_files": 3,
       "total_symbols": 15,
       "total_chunks": 15,
       "parse_errors": 0,
       "indexing_time": 2.5
     }
   }
   ```

3. **Search Results** - Relevant code snippets for each query:
   - Query: "calculator add numbers" returns the `Calculator.add()` method
   - Query: "load configuration from JSON file" returns the `load_config()` function
   - Query: "user data model" returns the `User` dataclass

4. **Incremental Update** - Shows how CodeGrok detects file changes and only re-indexes modified files

5. **Load-Only Mode** - Demonstrates reloading a previously created index from disk

### Example Session

```
╭─────────────────────────────────────────╮
│      CodeGrok MCP Demo                  │
│ Testing all MCP tools with real output  │
╰─────────────────────────────────────────╯

Project path: /tmp/codegrok_demo_xyz/sample_project

>>> Calling: list_supported_languages

┌─────────────────────────────────────────┐
│ Supported Languages                      │
├───────────────┬─────────────────────────┤
│ Language      │ Extensions              │
├───────────────┼─────────────────────────┤
│ C             │ .c, .h                  │
│ C++           │ .cpp, .hpp, .cc, .hh    │
│ Go            │ .go                     │
│ JavaScript    │ .js, .jsx               │
│ Python        │ .py                     │
│ Rust          │ .rs                     │
│ TypeScript    │ .ts, .tsx               │
└───────────────┴─────────────────────────┘

>>> Calling: learn
Parameters:
{
  "path": "/tmp/codegrok_demo_xyz/sample_project",
  "mode": "auto"
}

Result:
{
  "status": "success",
  "message": "Indexed 3 files with 15 symbols (15 chunks)",
  "stats": {
    "total_files": 3,
    "total_symbols": 15,
    "total_chunks": 15,
    "parse_errors": 0,
    "indexing_time": 2.34
  }
}

[... more output follows ...]
```

### Troubleshooting

**Issue: Import errors**
```
ModuleNotFoundError: No module named 'codegrok_mcp'
```
**Solution**: Install the package with `pip install -e .` from the project root.

**Issue: Permission denied**
```
PermissionError: [Errno 13] Permission denied: '/path/to/project'
```
**Solution**: Ensure you have read/write access to the project directory.

**Issue: Out of memory during indexing**
```
MemoryError or process killed
```
**Solution**: The project may be too large. Try with a smaller project or increase available RAM.

## Understanding the Output

### JSON Results

All tool calls return JSON responses. Key fields:

- `status`: "success" or "error"
- `message`: Human-readable description
- `stats`: Indexing statistics (files, symbols, chunks, errors, time)
- `sources`: Array of code snippets (for `get_sources`)

### Color Coding

- **Cyan**: Tool names and headers
- **Yellow**: Parameters and warnings
- **Green**: Successful results
- **Red**: Errors
- **Dim**: Metadata and less important information

## Technical Details

### How It Works

The demo script:

1. Imports MCP tool wrappers from `codegrok_mcp.mcp.server`
2. Extracts the underlying `.fn` functions (bypassing FastMCP protocol)
3. Calls these functions directly with appropriate parameters
4. Formats and displays the JSON responses using Rich library

### What Gets Created

Running the demo creates:

- `.codegrok/` directory in the indexed project
  - `chromadb/` - Vector database with embeddings
  - `metadata.json` - File hashes and indexing metadata

These files persist between runs and enable incremental reindexing.

## Adding More Tests

To add additional test scenarios:

1. Create a new Python script in this directory
2. Import the MCP tool functions:
   ```python
   from codegrok_mcp.mcp.server import learn, get_sources, get_stats
   learn_fn = learn.fn
   get_sources_fn = get_sources.fn
   get_stats_fn = get_stats.fn
   ```
3. Call the functions with test data
4. Verify the results

## Related Documentation

- **User Guide**: `../README.md` - How to use CodeGrok MCP with MCP clients
- **Developer Guide**: `../CLAUDE.md` - Comprehensive codebase documentation
- **Implementation Plan**: `../docs/IMPLEMENTATION_PLAN.md` - Detailed feature roadmap

## Support

If you encounter issues:

1. Verify installation: `pip show codegrok-mcp`
2. Check Python version: `python --version` (requires 3.10+)
3. Review error messages in the output
4. Consult `CLAUDE.md` for troubleshooting tips
