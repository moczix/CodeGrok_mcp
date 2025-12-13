"""
Integration tests for SourceRetriever - the core indexing/search engine.
These test the actual functionality without MCP overhead.
"""
import pytest
import tempfile
from pathlib import Path
from codegrok_mcp.indexing.source_retriever import SourceRetriever


from unittest.mock import patch

class TestSourceRetrieverIndexing:
    """Test indexing functionality."""

    def test_index_python_project(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )

            retriever.index_codebase()
            stats = retriever.get_stats()

            assert stats["total_files"] >= 2
            assert stats["total_symbols"] > 0

    def test_index_respects_file_extensions(self, multi_lang_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(multi_lang_project),
                persist_path=persist_dir
            )

            # Index only Python files
            retriever.index_codebase(file_extensions=[".py"])
            stats = retriever.get_stats()

            assert stats["total_files"] == 1  # Only app.py

    def test_index_creates_persist_directory(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            persist_path = Path(persist_dir) / "chroma"
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=str(persist_path)
            )

            retriever.index_codebase()

            assert persist_path.exists()


class TestSourceRetrieverSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def indexed_retriever(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )
            retriever.index_codebase()
            yield retriever

    def test_get_sources_before_indexing(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )
            results, _ = retriever.get_sources_for_question("test")
            assert results == []

    def test_search_returns_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question(
            "calculator",
            n_results=5
        )

        assert len(results) > 0

    def test_search_respects_n_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question(
            "function",
            n_results=2
        )

        assert len(results) <= 2

    def test_search_returns_relevant_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question(
            "add numbers calculator",
            n_results=5
        )

        # Should find calculator-related code
        result_text = " ".join([str(r) for r in results])
        assert "add" in result_text.lower() or "calculator" in result_text.lower()


class TestIncrementalReindex:
    """Test incremental reindexing."""

    def test_detects_modified_files(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )

            # Initial index
            retriever.index_codebase()

            # Modify a file
            main_py = temp_project / "main.py"
            main_py.write_text('def brand_new_function(): pass')

            # Incremental reindex
            retriever.incremental_reindex()

            # Search should find the new function
            results, _ = retriever.get_sources_for_question("brand_new_function")
            result_text = " ".join([str(r) for r in results])
            assert "brand_new" in result_text.lower()

    def test_handles_new_files(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )

            # Initial index
            retriever.index_codebase()

            # Add a new file with unique content
            new_file = temp_project / "new_module.py"
            new_file.write_text('def unique_xyz_function_12345(): pass')

            # Incremental reindex
            retriever.incremental_reindex()

            # Verify the new function is searchable
            results, _ = retriever.get_sources_for_question("unique_xyz_function_12345")
            result_text = " ".join([str(r) for r in results])
            assert "unique_xyz_function_12345" in result_text.lower()

    def test_incremental_reindex_with_parse_error(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir
            )
            retriever.index_codebase()
            
            # Create broken file
            (temp_project / "broken.py").write_text("def (")
            
            # Should not raise exception
            result = retriever.incremental_reindex()
            
            # Check result properties if possible, or just sufficient that it didn't crash
            assert result is not None
            assert result['files_added'] == 1 or result['files_modified'] == 1

    def test_index_codebase_with_parse_error(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project),
                persist_path=persist_dir,
                parallel=False
            )
            
            with patch.object(retriever.parser, 'parse_file', side_effect=Exception("Boom")):
                retriever.index_codebase()
                # Should handle error and continue/finish
                assert retriever.stats['parse_errors'] > 0

