# Security Audit Report: CodeGrok MCP

**Audit Date**: December 15, 2025
**Auditor**: Claude Security Code Auditor
**Codebase**: CodeGrok MCP v0.1.0
**Overall Risk Rating**: **MEDIUM-HIGH**

---

## Executive Summary

CodeGrok MCP is an MCP server for semantic code search. This security audit identified **12 security findings** across the codebase, including **3 HIGH severity issues** that could allow path traversal attacks, arbitrary file reads, and denial of service conditions.

The most critical concerns are:
1. **Insufficient path validation** allowing access to sensitive files outside intended directories
2. **Unbounded resource consumption** enabling denial of service attacks
3. **Trusting remote code** in embedding model loading

No critical (exploitable RCE) vulnerabilities were found, but the HIGH severity issues should be addressed before production deployment.

---

## Finding Summary

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 0 | None identified |
| HIGH | 3 | Path Traversal, Arbitrary File Read, Resource Exhaustion |
| MEDIUM | 5 | Remote Code Trust, JSON Deserialization, Info Disclosure, Input Validation, Thread Safety |
| LOW | 3 | Symlink Following, Rate Limiting, Verbose Logging |
| INFO | 1 | Security Best Practices |

---

## Detailed Findings

### [HIGH-001] Path Traversal in `learn` Tool

**Severity**: HIGH
**CVSS Score**: 7.5
**Location**: `src/codegrok_mcp/mcp/server.py:141-170`

**Description**:
The `learn` MCP tool accepts arbitrary file paths without validating they are within allowed directories. An attacker can index sensitive system directories.

**Vulnerable Code**:
```python
@mcp.tool
async def learn(project_path: str, project_name: str | None = None, ...) -> str:
    # No validation that project_path is in an allowed directory
    resolved_path = Path(project_path).resolve()
    # ...proceeds to index all files in path
```

**Attack Scenario**:
```python
# Attacker calls learn tool with sensitive path
await learn(project_path="/etc/passwd", project_name="secrets")
await learn(project_path="/home/user/.ssh", project_name="ssh_keys")
```

**Remediation**:
```python
ALLOWED_BASE_PATHS = [Path.home() / "projects", Path("/var/www")]

def validate_project_path(project_path: str) -> Path:
    resolved = Path(project_path).resolve()
    if not any(resolved.is_relative_to(base) for base in ALLOWED_BASE_PATHS):
        raise ValueError(f"Path {project_path} is outside allowed directories")
    return resolved
```

---

### [HIGH-002] Arbitrary File Read via Parser

**Severity**: HIGH
**CVSS Score**: 7.1
**Location**: `src/codegrok_mcp/parsers/treesitter_parser.py:99-191`

**Description**:
The Tree-sitter parser reads any file with supported extensions without path validation, allowing information disclosure from sensitive files.

**Vulnerable Code**:
```python
def parse_file(self, file_path: str | Path) -> ParsedFile:
    file_path = Path(file_path)
    # No validation - reads any file
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
```

**Risk**:
Sensitive configuration files (`.env`, `secrets.py`, `config.json`) could be read and indexed, exposing credentials.

**Remediation**:
1. Add path validation before file read
2. Implement a blocklist of sensitive file patterns:
```python
SENSITIVE_PATTERNS = [".env", "secrets", "credentials", ".pem", ".key"]

def is_sensitive_file(path: Path) -> bool:
    return any(pattern in path.name.lower() for pattern in SENSITIVE_PATTERNS)
```

---

### [HIGH-003] Unbounded Resource Consumption (DoS)

**Severity**: HIGH
**CVSS Score**: 7.5
**Location**: `src/codegrok_mcp/indexing/source_retriever.py:289-518`

**Description**:
No limits on file count, total size, chunk count, or memory usage during indexing operations.

**Vulnerable Code**:
```python
def index_codebase(self, ...):
    # Discovers ALL files without limit
    files_to_index = self._discover_files()  # Could be millions

    # No limit on total memory consumption
    for file_path in files_to_index:
        self._process_file(file_path)  # Each file adds to memory
```

**Attack Scenario**:
```python
# Point to a huge directory
await learn(project_path="/", project_name="everything")  # Indexes entire system
```

**Remediation**:
```python
MAX_FILES = 50000
MAX_TOTAL_SIZE_MB = 500
MAX_CHUNKS = 100000

def index_codebase(self, ...):
    files = self._discover_files()
    if len(files) > MAX_FILES:
        raise ResourceLimitError(f"Too many files: {len(files)} > {MAX_FILES}")

    total_size = sum(f.stat().st_size for f in files)
    if total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
        raise ResourceLimitError(f"Total size exceeds {MAX_TOTAL_SIZE_MB}MB")
```

---

### [MEDIUM-004] Trust Remote Code in Embedding Model

**Severity**: MEDIUM
**CVSS Score**: 6.8
**Location**: `src/codegrok_mcp/indexing/embedding_service.py:64-116`

**Description**:
The `trust_remote_code=True` flag allows arbitrary code execution when loading Nomic embedding models.

**Vulnerable Code**:
```python
self.model = SentenceTransformer(
    model_name,
    trust_remote_code=True,  # DANGEROUS: Allows arbitrary code execution
    device=self.device
)
```

**Risk**:
If the model repository on HuggingFace is compromised, arbitrary code could execute during model loading.

**Remediation**:
```python
# Use only verified models that don't require trust_remote_code
SAFE_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

if model_name in SAFE_MODELS:
    self.model = SentenceTransformer(model_name, device=self.device)
else:
    raise ValueError(f"Model {model_name} requires trust_remote_code, which is disabled")
```

---

### [MEDIUM-005] Insecure JSON Deserialization

**Severity**: MEDIUM
**CVSS Score**: 5.3
**Location**: `src/codegrok_mcp/indexing/source_retriever.py:632-653`

**Description**:
Metadata is loaded from JSON files without schema validation, potentially allowing injection of malicious data.

**Vulnerable Code**:
```python
def load_existing_index(self):
    metadata_path = self.codegrok_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)  # No validation

    self.project_path = metadata.get("project_path")  # Directly used
```

**Remediation**:
```python
from pydantic import BaseModel, validator

class IndexMetadata(BaseModel):
    project_name: str
    project_path: str
    file_hashes: dict[str, str]
    chunk_size: int
    indexed_at: str

    @validator("project_path")
    def validate_path(cls, v):
        if ".." in v or not Path(v).exists():
            raise ValueError("Invalid project path")
        return v

# Use validated model
metadata = IndexMetadata.parse_file(metadata_path)
```

---

### [MEDIUM-006] Information Disclosure in Error Messages

**Severity**: MEDIUM
**CVSS Score**: 4.3
**Location**: Multiple files

**Description**:
Error messages expose full file paths and internal implementation details.

**Examples**:
```python
# server.py:158
return f"Error indexing codebase: {str(e)}"  # Exposes internal error details

# source_retriever.py:398
logger.error(f"Failed to parse {file_path}: {e}")  # Full path in logs
```

**Remediation**:
```python
class SafeError(Exception):
    """User-safe error message"""
    def __init__(self, user_message: str, internal_message: str):
        self.user_message = user_message
        self.internal_message = internal_message
        super().__init__(user_message)

# Use generic messages for users
try:
    # operation
except Exception as e:
    logger.error(f"Internal error: {e}")  # Log full details
    raise SafeError("Operation failed. Please check logs.", str(e))
```

---

### [MEDIUM-007] Missing Input Validation on Query Parameters

**Severity**: MEDIUM
**CVSS Score**: 5.0
**Location**: `src/codegrok_mcp/mcp/server.py:324-364`

**Description**:
No length limits on search queries or validation of filter parameters.

**Vulnerable Code**:
```python
@mcp.tool
async def get_sources(
    query: str,  # No length limit - could be megabytes
    n_results: int = 10,  # No upper bound validated
    language: str | None = None,  # Not validated against known languages
    symbol_type: str | None = None  # Not validated against SymbolType enum
) -> str:
```

**Attack Scenario**:
```python
# Memory exhaustion with huge query
await get_sources(query="A" * 10_000_000, n_results=1000000)
```

**Remediation**:
```python
MAX_QUERY_LENGTH = 10000
MAX_N_RESULTS = 100
VALID_LANGUAGES = ["python", "javascript", "typescript", ...]
VALID_SYMBOL_TYPES = [e.value for e in SymbolType]

@mcp.tool
async def get_sources(query: str, n_results: int = 10, ...) -> str:
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long (max {MAX_QUERY_LENGTH} chars)"
    if n_results > MAX_N_RESULTS:
        n_results = MAX_N_RESULTS
    if language and language not in VALID_LANGUAGES:
        return f"Unknown language: {language}"
```

---

### [MEDIUM-008] Global Singleton State Without Thread Safety

**Severity**: MEDIUM
**CVSS Score**: 4.7
**Location**: `src/codegrok_mcp/mcp/state.py:21-35`

**Description**:
Global state accessed without synchronization, potential race conditions.

**Vulnerable Code**:
```python
_current_chat: SourceRetriever | None = None

def get_chat() -> SourceRetriever | None:
    return _current_chat  # No lock

def set_chat(chat: SourceRetriever | None) -> None:
    global _current_chat
    _current_chat = chat  # Race condition possible
```

**Remediation**:
```python
import threading

_state_lock = threading.Lock()
_current_chat: SourceRetriever | None = None

def get_chat() -> SourceRetriever | None:
    with _state_lock:
        return _current_chat

def set_chat(chat: SourceRetriever | None) -> None:
    global _current_chat
    with _state_lock:
        _current_chat = chat
```

---

### [LOW-009] Symlink Following Without Validation

**Severity**: LOW
**CVSS Score**: 3.7
**Location**: `src/codegrok_mcp/parsers/treesitter_parser.py:210-250`

**Description**:
Directory traversal follows symlinks without checking if they point outside the project.

**Remediation**:
```python
def is_safe_symlink(path: Path, base_dir: Path) -> bool:
    """Check symlink doesn't escape base directory"""
    if path.is_symlink():
        target = path.resolve()
        return target.is_relative_to(base_dir)
    return True
```

---

### [LOW-010] No Rate Limiting on MCP Tools

**Severity**: LOW
**CVSS Score**: 3.1
**Location**: `src/codegrok_mcp/mcp/server.py`

**Description**:
No rate limiting on expensive operations like `learn` and `get_sources`.

**Remediation**:
```python
from functools import wraps
import time

_last_call: dict[str, float] = {}
RATE_LIMITS = {"learn": 60, "get_sources": 1}  # seconds

def rate_limit(tool_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            if tool_name in _last_call:
                elapsed = now - _last_call[tool_name]
                if elapsed < RATE_LIMITS.get(tool_name, 0):
                    return f"Rate limited. Wait {RATE_LIMITS[tool_name] - elapsed:.0f}s"
            _last_call[tool_name] = now
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

---

### [LOW-011] Verbose Logging May Expose Sensitive Data

**Severity**: LOW
**CVSS Score**: 2.7
**Location**: Multiple files

**Description**:
Debug logging includes file contents and paths that may contain secrets.

**Remediation**:
- Implement log sanitization
- Use structured logging with sensitive field filtering
- Disable debug logging in production

---

### [INFO-012] Security Best Practices Observations

**Positive Findings**:
- Type hints throughout codebase improve code safety
- Frozen dataclasses prevent accidental mutation
- Binary file detection prevents parsing issues
- `.git` and `node_modules` directories are excluded

**Recommendations for Improvement**:
- Add security headers to responses
- Implement audit logging for tool invocations
- Add configuration file for security settings
- Consider sandboxing file operations

---

## Remediation Roadmap

### Phase 1: Immediate
| Finding | Action | Effort |
|---------|--------|--------|
| HIGH-001 | Add path validation to `learn` tool | 2 hours |
| HIGH-002 | Add file blocklist to parser | 2 hours |
| HIGH-003 | Implement resource limits | 4 hours |

### Phase 2: Short-term
| Finding | Action | Effort |
|---------|--------|--------|
| MEDIUM-004 | Remove `trust_remote_code` or whitelist models | 1 hour |
| MEDIUM-005 | Add JSON schema validation | 3 hours |
| MEDIUM-006 | Sanitize error messages | 2 hours |
| MEDIUM-007 | Add input validation | 2 hours |
| MEDIUM-008 | Add thread-safe state management | 2 hours |

### Phase 3: Medium-term
| Finding | Action | Effort |
|---------|--------|--------|
| LOW-009 | Add symlink validation | 1 hour |
| LOW-010 | Implement rate limiting | 3 hours |
| LOW-011 | Implement log sanitization | 2 hours |

---

## Files Requiring Changes

1. **`src/codegrok_mcp/mcp/server.py`** - Path validation, input validation, rate limiting
2. **`src/codegrok_mcp/parsers/treesitter_parser.py`** - File blocklist, symlink handling
3. **`src/codegrok_mcp/indexing/source_retriever.py`** - Resource limits, JSON validation
4. **`src/codegrok_mcp/indexing/embedding_service.py`** - Remove trust_remote_code
5. **`src/codegrok_mcp/mcp/state.py`** - Thread safety
6. **`src/codegrok_mcp/core/exceptions.py`** (new) - Custom security exceptions

---

## Conclusion

The CodeGrok MCP codebase has a **MEDIUM-HIGH** risk profile primarily due to insufficient input validation and resource consumption controls. The identified issues are common in code indexing tools but should be addressed before production deployment.

**Priority Actions**:
1. Implement path validation to prevent traversal attacks
2. Add resource limits to prevent DoS
3. Remove `trust_remote_code` from embedding service

The codebase shows good foundational practices (type hints, error handling, modular design) that will facilitate implementing these security improvements.

---

*Report generated by Claude Security Code Auditor*
