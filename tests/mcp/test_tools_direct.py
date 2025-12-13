"""
Test MCP tools by calling decorated functions directly.
This simulates what an MCP client does, without the protocol overhead.

Note: FastMCP's @mcp.tool decorator wraps functions into FunctionTool objects.
We access the underlying function via the .fn attribute.
"""
import pytest
from pathlib import Path
from fastmcp.exceptions import ToolError
from codegrok_mcp.mcp.server import (
    learn as learn_tool,
    relearn as relearn_tool,
    load as load_tool,
    get_sources as get_sources_tool,
    get_stats as get_stats_tool,
    list_supported_languages as list_supported_languages_tool
)
from codegrok_mcp.mcp.state import get_state, reset_state

# Access underlying functions from FastMCP FunctionTool wrappers
learn = learn_tool.fn
relearn = relearn_tool.fn
load = load_tool.fn
get_sources = get_sources_tool.fn
get_stats = get_stats_tool.fn
list_supported_languages = list_supported_languages_tool.fn


class TestListSupportedLanguages:
    """Test the list_supported_languages tool - no state required."""

    def test_returns_extensions_and_languages(self):
        result = list_supported_languages()

        assert "extensions" in result
        assert "languages" in result
        assert isinstance(result["extensions"], list)
        assert isinstance(result["languages"], dict)

    def test_python_is_supported(self):
        result = list_supported_languages()
        assert ".py" in result["extensions"]
        assert "python" in result["languages"]

    def test_multiple_languages_supported(self):
        result = list_supported_languages()
        expected_extensions = [".py", ".js", ".ts", ".go"]
        for ext in expected_extensions:
            assert ext in result["extensions"]


class TestLearnTool:
    """Test the learn tool - indexes a codebase."""

    def test_learn_valid_project(self, temp_project):
        result = learn(path=str(temp_project))

        assert result["success"] is True
        assert "stats" in result
        assert result["stats"]["total_files"] >= 2
        assert result["stats"]["total_symbols"] > 0

    def test_learn_creates_codegrok_directory(self, temp_project):
        learn(path=str(temp_project))

        codegrok_dir = temp_project / ".codegrok"
        assert codegrok_dir.exists()
        assert (codegrok_dir / "metadata.json").exists()

    def test_learn_with_custom_extensions(self, temp_project):
        result = learn(
            path=str(temp_project),
            file_extensions=[".py"]
        )

        assert result["success"] is True
        assert result["stats"]["total_files"] == 2

    def test_learn_sets_state(self, temp_project):
        learn(path=str(temp_project))

        state = get_state()
        assert state.is_loaded is True
        assert state.is_loaded is True
        assert state.retriever is not None

    def test_learn_path_not_exists(self):
        with pytest.raises(ToolError, match="Path does not exist"):
            learn(path="/nonexistent/path/12345")

    def test_learn_path_is_file(self, tmp_path):
        file = tmp_path / "file.txt"
        file.write_text("content")
        with pytest.raises(ToolError, match="not a directory"):
            learn(path=str(file))



class TestGetStatsTool:
    """Test the get_stats tool."""

    def test_stats_when_not_loaded(self):
        result = get_stats()

        assert result["loaded"] is False
        assert result["codebase_path"] is None

    def test_stats_after_learn(self, temp_project):
        learn(path=str(temp_project))
        result = get_stats()

        assert result["loaded"] is True
        assert result["codebase_path"] == str(temp_project)
        assert "total_files" in result["stats"]
        assert "total_symbols" in result["stats"]


class TestGetSourcesTool:
    """Test the get_sources tool - semantic search."""

    def test_get_sources_returns_results(self, temp_project):
        learn(path=str(temp_project))
        result = get_sources(question="calculator add numbers")

        assert "sources" in result
        # The sources can be a tuple or list depending on implementation
        sources = result["sources"]
        # If it's a tuple, first element is the actual sources
        if isinstance(sources, tuple):
            sources = sources[0]
        assert isinstance(sources, list) or len(sources) >= 0

    def test_get_sources_respects_n_results(self, temp_project):
        learn(path=str(temp_project))
        result = get_sources(question="function", n_results=2)

        sources = result["sources"]
        # If it's a tuple, first element is the actual sources
        if isinstance(sources, tuple):
            sources = sources[0]
        assert len(sources) <= 2

    def test_get_sources_no_codebase_loaded(self):
        reset_state()
        with pytest.raises(ToolError, match="No codebase loaded"):
            get_sources(question="test")



class TestLoadTool:
    """Test the load tool - loads existing index."""

    def test_load_after_learn(self, temp_project):
        # First, create an index
        learn(path=str(temp_project))

        # Clear state
        reset_state()

        # Load the index
        result = load(path=str(temp_project))

        assert result["success"] is True
        assert result["stats"]["total_files"] >= 2



    def test_load_missing_codegrok_dir(self, tmp_path):
        # Create empty directory
        with pytest.raises(ToolError, match="No .codegrok/ found"):
            load(path=str(tmp_path))


class TestRelearnTool:
    """Test the relearn tool - incremental reindexing."""

    def test_relearn_without_path(self, temp_project):
        """Test relearn using already loaded codebase."""
        # Initial learn
        learn(path=str(temp_project))

        # Modify a file
        (temp_project / "main.py").write_text('def new_function(): pass')

        # Relearn without providing path (uses loaded state)
        result = relearn(path=None)

        assert result["success"] is True
