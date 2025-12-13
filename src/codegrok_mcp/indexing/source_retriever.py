"""
Source retriever for semantic code search.

This module provides semantic search over code using embeddings with persistent
storage. It handles indexing, storage, and retrieval of code chunks for RAG
(Retrieval-Augmented Generation) applications.

Note: This module is LLM-agnostic. It provides source retrieval only.
LLM integration is handled by the MCP layer.

Usage:
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    # Learn mode (with persistence)
    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )
    retriever.index_codebase()
    retriever.save_metadata("/path/to/code/.codegrok/metadata.json")

    # Retrieval mode (load existing)
    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )
    retriever.load_existing_index()
    sources = retriever.get_sources_for_question("How does authentication work?")
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Install with: pip install chromadb")

# Import native embedding service
from codegrok_mcp.indexing.embedding_service import get_embedding_service, EmbeddingService

from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser
from codegrok_mcp.parsers.language_configs import get_supported_extensions
from codegrok_mcp.core.models import Symbol, SymbolType


# Derived from authoritative EXTENSION_MAP in language_configs.py (30+ extensions, 9 languages)
# This eliminates duplication and ensures extensions stay in sync
SUPPORTED_EXTENSIONS = list(get_supported_extensions())


def count_codebase_files(codebase_path: Path) -> int:
    """Quick file count for ETA estimation.

    Args:
        codebase_path: Path to the codebase root directory.

    Returns:
        Total count of files with supported extensions.
    """
    count = 0
    for ext in SUPPORTED_EXTENSIONS:
        count += len(list(codebase_path.rglob(f"*{ext}")))
    return count


@dataclass
class CodeChunk:
    """A chunk of code suitable for embedding."""
    id: str
    text: str
    filepath: str
    symbol_name: str
    symbol_type: str
    line_start: int
    metadata: Dict[str, Any]


class SourceRetriever:
    """
    Source retriever for semantic code search.

    This class handles:
    - Phase 1: Parsing (using TreeSitterParser)
    - Phase 2: Storage (ChromaDB persistent/in-memory)
    - Phase 4: Semantic search (embeddings)

    Note: LLM integration is NOT handled here. This is a retrieval-only component.
    """

    def __init__(
        self,
        codebase_path: str,
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "codebase",
        verbose: bool = True,
        persist_path: Optional[str] = None,

        # Parallel indexing options (3-5x faster for large codebases)
        parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the source retriever.

        Args:
            codebase_path: Path to codebase root directory
            embedding_model: Embedding model name (native or Ollama)
            collection_name: ChromaDB collection name
            verbose: Print progress messages
            persist_path: Path for ChromaDB persistent storage (None = in-memory)
            use_native_embedding: Use native SentenceTransformers (True) or Ollama API (False)
            parallel: Enable parallel file parsing for faster indexing (3-5x speedup)
            max_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.codebase_path = Path(codebase_path)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.verbose = verbose
        self.persist_path = persist_path

        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize parser
        self.parser = TreeSitterParser()

        # Initialize embedding service
        self._log(f"Using native embedding: {embedding_model}")
        self.embedding_service = get_embedding_service(
            embedding_model,
            show_progress=verbose  # Only show tqdm progress bar if verbose
        )

        # Initialize ChromaDB (persistent or in-memory)
        if persist_path:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_path)
            self._log(f"Using persistent storage: {persist_path}")
        else:
            self.chroma_client = chromadb.Client()
            self._log("Using in-memory storage (data will be lost on exit)")

        self.collection = None

        # Statistics
        self.stats = {
            'total_files': 0,
            'total_symbols': 0,
            'total_chunks': 0,
            'parse_errors': 0,
            'indexing_time': 0.0
        }

        # Metadata storage for incremental reindexing (file modification times)
        self._metadata: Dict[str, Any] = {}

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)

    def _create_chunk_text(self, symbol: Symbol) -> str:
        """
        Create embedding text from a symbol.

        Format:
            # filepath:line
            type: name

            signature

            docstring

            code_snippet

            Imports: ...
            Calls: ...

        Args:
            symbol: Symbol to convert to chunk text

        Returns:
            Formatted text suitable for embedding
        """
        parts = [
            f"# {symbol.filepath}:{symbol.line_start}",
            f"{symbol.type.value}: {symbol.name}",
            "",
        ]

        if symbol.signature:
            parts.append(symbol.signature)
            parts.append("")

        if symbol.docstring:
            parts.append(symbol.docstring)
            parts.append("")

        if symbol.code_snippet:
            parts.append(symbol.code_snippet)
            parts.append("")

        if symbol.imports:
            imports_str = ", ".join(symbol.imports[:10])
            if len(symbol.imports) > 10:
                imports_str += f", ... ({len(symbol.imports) - 10} more)"
            parts.append(f"Imports: {imports_str}")

        if symbol.calls:
            calls_str = ", ".join(symbol.calls[:10])
            if len(symbol.calls) > 10:
                calls_str += f", ... ({len(symbol.calls) - 10} more)"
            parts.append(f"Calls: {calls_str}")

        return "\n".join(parts)

    def _create_chunk(self, symbol: Symbol) -> CodeChunk:
        """
        Create a CodeChunk from a Symbol.

        Args:
            symbol: Symbol to convert

        Returns:
            CodeChunk ready for embedding
        """
        chunk_id = f"{symbol.filepath}:{symbol.name}:{symbol.line_start}"
        chunk_text = self._create_chunk_text(symbol)

        return CodeChunk(
            id=chunk_id,
            text=chunk_text,
            filepath=symbol.filepath,
            symbol_name=symbol.name,
            symbol_type=symbol.type.value,
            line_start=symbol.line_start,
            metadata={
                'filepath': symbol.filepath,
                'name': symbol.name,
                'type': symbol.type.value,
                'line': symbol.line_start,
                'signature': symbol.signature,
                'parent': symbol.parent or ""
            }
        )

    def index_codebase(
        self,
        file_extensions: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Index the entire codebase.

        This is the one-time expensive operation that:
        1. Parses all files (Phase 1)
        2. Creates chunks (Phase 2)
        3. Generates embeddings (Phase 4)
        4. Stores in ChromaDB (Phase 2)

        Args:
            file_extensions: List of extensions to parse (default: all supported)
            progress_callback: Optional callback function for progress updates.
                Called with (event_type, data) where event_type is one of:
                - "files_found": data = {"files": List[Path], "codebase_path": Path}
                - "parsing_start": data = {"total": int}
                - "file_parsed": data = {"path": str, "symbols": int, "index": int, "total": int}
                - "parse_error": data = {"path": str, "error": str}
                - "chunks_created": data = {"total": int}
                - "embedding_start": data = {"total": int, "eta_minutes": float}
                - "embedding_progress": data = {"current": int, "total": int}
                - "complete": data = {"stats": dict}
        """
        if file_extensions is None:
            file_extensions = SUPPORTED_EXTENSIONS

        def emit(event_type: str, data: dict):
            """Emit progress event to callback if provided."""
            if progress_callback:
                progress_callback(event_type, data)

        # Legacy logging for when no callback provided
        if not progress_callback:
            self._log("\n" + "="*80)
            self._log("INDEXING CODEBASE")
            self._log("="*80)
            self._log(f"Codebase: {self.codebase_path}")
            self._log(f"Extensions: {file_extensions}")
            self._log(f"Embedding model: {self.embedding_model}")

        start_time = time.time()

        # Step 1: Find all files
        if not progress_callback:
            self._log("\nStep 1: Finding files...")

        all_files = []
        for ext in file_extensions:
            all_files.extend(self.codebase_path.rglob(f"*{ext}"))

        self.stats['total_files'] = len(all_files)

        # Store file modification times for incremental reindexing
        file_mtimes = {}
        for filepath in all_files:
            try:
                file_mtimes[str(filepath)] = filepath.stat().st_mtime
            except OSError:
                pass  # Skip files that can't be stat'd
        self._metadata['file_mtimes'] = file_mtimes

        emit("files_found", {
            "files": all_files,
            "codebase_path": self.codebase_path
        })

        if not progress_callback:
            self._log(f"Found {len(all_files)} files")

        # Step 2: Parse all files
        if not progress_callback:
            self._log("\nStep 2: Parsing files...")

        emit("parsing_start", {"total": len(all_files)})
        all_symbols = []

        # Use parallel parsing if enabled and there are enough files
        use_parallel = self.parallel and len(all_files) > 10
        if use_parallel:
            # Parallel parsing (3-5x faster for large codebases)
            from codegrok_mcp.indexing.parallel_indexer import parallel_parse_files

            if not progress_callback:
                self._log(f"  Using parallel parsing with {self.max_workers or 'auto'} workers...")

            all_symbols, parse_errors = parallel_parse_files(
                files=all_files,
                max_workers=self.max_workers,
                progress_callback=progress_callback
            )
            self.stats['parse_errors'] = parse_errors
        else:
            # Sequential parsing (original code)
            for i, filepath in enumerate(all_files, 1):
                symbols_count = 0
                try:
                    parsed = self.parser.parse_file(str(filepath))
                    all_symbols.extend(parsed.symbols)
                    symbols_count = len(parsed.symbols)

                    emit("file_parsed", {
                        "path": str(filepath),
                        "symbols": symbols_count,
                        "index": i,
                        "total": len(all_files)
                    })
                except Exception as e:
                    self.stats['parse_errors'] += 1
                    emit("parse_error", {"path": str(filepath), "error": str(e)})
                    if not progress_callback and self.verbose and self.stats['parse_errors'] <= 5:
                        self._log(f"  Error parsing {filepath}: {e}")

                # Legacy progress for no callback
                if not progress_callback and self.verbose and i % 100 == 0:
                    print(f"  Parsed {i}/{len(all_files)} files...", end='\r')

        self.stats['total_symbols'] = len(all_symbols)
        if not progress_callback:
            self._log(f"\nParsed {len(all_symbols):,} symbols from {len(all_files)} files")

        # Step 3: Create chunks
        if not progress_callback:
            self._log("\nStep 3: Creating chunks...")

        chunks = [self._create_chunk(symbol) for symbol in all_symbols]
        self.stats['total_chunks'] = len(chunks)

        emit("chunks_created", {"total": len(chunks)})

        if not progress_callback:
            self._log(f"Created {len(chunks):,} chunks")

        # Step 4: Create ChromaDB collection
        if not progress_callback:
            self._log("\nStep 4: Creating vector database...")

        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": f"Code embeddings for {self.codebase_path.name}"}
        )

        # Step 5: Generate embeddings and store
        eta_minutes = len(chunks) / 50 / 60  # ~50 embeddings/sec native

        emit("embedding_start", {"total": len(chunks), "eta_minutes": eta_minutes})

        if not progress_callback:
            self._log(f"\nStep 5: Generating embeddings (ETA: ~{eta_minutes:.1f} minutes)...")
            self._log("(You can interrupt and resume later)")

        batch_size = 100
        embedding_start_time = time.time()
        chunks_per_second = None  # Will be calibrated after first batch

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            current_count = i + len(batch)
            elapsed = time.time() - embedding_start_time

            # Calibrate speed after first batch, then update continuously
            if elapsed > 0 and current_count > 0:
                chunks_per_second = current_count / elapsed

            # Calculate remaining time estimate
            remaining_seconds = None
            if chunks_per_second and chunks_per_second > 0:
                remaining_chunks = len(chunks) - current_count
                remaining_seconds = remaining_chunks / chunks_per_second

            emit("embedding_progress", {
                "current": current_count,
                "total": len(chunks),
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining_seconds,
                "chunks_per_second": chunks_per_second
            })

            # Legacy progress for no callback
            if not progress_callback and self.verbose and i % 500 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(chunks) - i) / rate if rate > 0 else 0
                print(f"  Embedded {i:,}/{len(chunks):,} chunks "
                      f"({i/len(chunks)*100:.1f}%) "
                      f"- ETA: {remaining/60:.1f}m", end='\r')

            try:
                # Generate embeddings for batch
                texts = [chunk.text for chunk in batch]

                # Generate embeddings for batch
                texts = [chunk.text for chunk in batch]

                # Native batch embedding (10-20x faster)
                embeddings = self.embedding_service.embed_batch(texts)

                # Add to ChromaDB
                self.collection.add(
                    ids=[chunk.id for chunk in batch],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch]
                )
            except Exception as e:
                if not progress_callback:
                    self._log(f"\n  Error embedding batch {i}: {e}")
                continue

        self.stats['indexing_time'] = time.time() - start_time

        emit("complete", {"stats": self.stats.copy()})

        # Legacy summary for no callback
        if not progress_callback:
            self._log("\n\n" + "="*80)
            self._log("INDEXING COMPLETE")
            self._log("="*80)
            self._log(f"Files parsed:      {self.stats['total_files']:,}")
            self._log(f"Symbols extracted: {self.stats['total_symbols']:,}")
            self._log(f"Chunks created:    {self.stats['total_chunks']:,}")
            self._log(f"Parse errors:      {self.stats['parse_errors']}")
            self._log(f"Time elapsed:      {self.stats['indexing_time']:.1f}s ({self.stats['indexing_time']/60:.1f}m)")
            self._log(f"Ready for retrieval!")
            self._log("="*80 + "\n")

    def get_sources_for_question(
        self,
        question: str,
        n_results: int = 5
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Get source references and documents for a question.

        This performs semantic search to find relevant code chunks.

        Args:
            question: The question to search for
            n_results: Number of results to retrieve

        Returns:
            Tuple of (documents, sources) where:
                - documents: List of document texts with metadata
                - sources: List of formatted source strings for display
        """
        if self.collection is None:
            return [], []

        # Embed the question
        query_embedding = self.embedding_service.embed(question, is_query=True)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        # Format sources for display
        sources = []
        for metadata in metadatas:
            filepath = metadata['filepath']
            try:
                filepath = str(Path(filepath).relative_to(self.codebase_path))
            except ValueError:
                pass
            sources.append(f"{filepath}:{metadata['line']} - {metadata['name']}()")

        # Build document list with metadata
        doc_results = []
        for doc, metadata in zip(documents, metadatas):
            doc_results.append({
                'text': doc,
                'metadata': metadata
            })

        return doc_results, sources

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return self.stats.copy()

    def load_existing_index(self) -> bool:
        """
        Load an existing indexed collection from persistent storage.

        Returns:
            True if collection was loaded successfully, False otherwise
        """
        if not self.persist_path:
            self._log("No persistent storage configured")
            return False

        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            count = self.collection.count()
            self._log(f"Loaded existing index with {count:,} chunks")
            return True
        except Exception as e:
            self._log(f"No existing index found: {e}")
            return False

    def save_metadata(self, metadata_path: str) -> None:
        """
        Save indexing metadata to JSON file.

        Args:
            metadata_path: Path to save metadata JSON
        """
        metadata = {
            'codebase_path': str(self.codebase_path),
            'embedding_model': self.embedding_model,
            'collection_name': self.collection_name,
            'indexed_at': datetime.now().isoformat(),
            'stats': self.stats,
            'file_mtimes': self._metadata.get('file_mtimes', {})
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self._log(f"Metadata saved to {metadata_path}")

    def load_metadata(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """
        Load indexing metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Restore stats
            if 'stats' in metadata:
                self.stats = metadata['stats']
            # Restore file modification times for incremental reindexing
            if 'file_mtimes' in metadata:
                self._metadata['file_mtimes'] = metadata['file_mtimes']
            return metadata
        except FileNotFoundError:
            return None

    def incremental_reindex(
        self,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Re-index only files that changed since last indexing.

        This method provides efficient incremental updates by:
        1. Comparing file modification times (mtime) against stored metadata
        2. Identifying new, modified, and deleted files
        3. Removing stale chunks from ChromaDB for modified/deleted files
        4. Parsing and indexing only new/modified files
        5. Updating metadata with new mtimes

        Args:
            file_extensions: List of extensions to consider (default: all supported)

        Returns:
            Dict with keys:
                - files_added: Number of new files indexed
                - files_modified: Number of modified files re-indexed
                - files_deleted: Number of deleted files removed
                - chunks_added: Number of new chunks added to ChromaDB
                - chunks_removed: Number of stale chunks removed from ChromaDB
                - time_seconds: Total time taken for incremental reindex
        """
        start_time = time.time()

        extensions = file_extensions or SUPPORTED_EXTENSIONS

        # 1. Get stored file_mtimes from metadata
        stored_mtimes = self._metadata.get('file_mtimes', {})

        # 2. Scan current files and collect modification times
        # Skip common non-source directories
        skip_dirs = ['.git', 'node_modules', '__pycache__', '.codegrok', 'venv', '.venv']
        current_files: Dict[str, float] = {}

        for ext in extensions:
            for filepath in Path(self.codebase_path).rglob(f'*{ext}'):
                # Skip files in excluded directories
                if not any(skip_dir in str(filepath) for skip_dir in skip_dirs):
                    try:
                        current_files[str(filepath)] = filepath.stat().st_mtime
                    except OSError:
                        pass  # Skip files that can't be stat'd

        # 3. Categorize changes by comparing current vs stored
        stored_paths = set(stored_mtimes.keys())
        current_paths = set(current_files.keys())

        new_files = current_paths - stored_paths
        deleted_files = stored_paths - current_paths
        modified_files = {
            p for p in (current_paths & stored_paths)
            if current_files[p] > stored_mtimes.get(p, 0)
        }

        files_to_reindex = new_files | modified_files
        files_to_remove = deleted_files | modified_files

        self._log(f"Incremental reindex: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted")

        chunks_removed = 0
        chunks_added = 0

        # 4. Remove stale chunks from ChromaDB for modified/deleted files
        if files_to_remove and self.collection:
            for filepath in files_to_remove:
                try:
                    # ChromaDB delete by metadata filter
                    self.collection.delete(where={"filepath": filepath})
                    chunks_removed += 1
                except Exception as e:
                    if self.verbose:
                        self._log(f"Warning: Could not remove chunks for {filepath}: {e}")

        # 5. Parse and index new/modified files
        if files_to_reindex:
            chunks = []
            for filepath in files_to_reindex:
                try:
                    parsed = self.parser.parse_file(filepath)
                    for symbol in parsed.symbols:
                        chunk = self._create_chunk(symbol)
                        chunks.append(chunk)
                except Exception as e:
                    if self.verbose:
                        self._log(f"Warning: Could not parse {filepath}: {e}")
                    continue

            if chunks:
                # Generate embeddings for new chunks
                texts = [chunk.text for chunk in chunks]

                embeddings = self.embedding_service.embed_batch(texts)

                # Add to ChromaDB
                self.collection.add(
                    ids=[chunk.id for chunk in chunks],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in chunks],
                    metadatas=[chunk.metadata for chunk in chunks]
                )
                chunks_added = len(chunks)

        # 6. Update metadata with new file modification times
        self._metadata['file_mtimes'] = current_files

        # Persist metadata if we have a persist path
        if self.persist_path:
            metadata_path = Path(self.persist_path).parent / 'metadata.json'
            self.save_metadata(str(metadata_path))

        elapsed_time = round(time.time() - start_time, 2)

        result = {
            "files_added": len(new_files),
            "files_modified": len(modified_files),
            "files_deleted": len(deleted_files),
            "chunks_added": chunks_added,
            "chunks_removed": chunks_removed,
            "time_seconds": elapsed_time
        }

        self._log(f"Incremental reindex complete in {elapsed_time}s: "
                  f"+{chunks_added} chunks, -{chunks_removed} files processed")

        return result
