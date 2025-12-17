#!/usr/bin/env python3
"""
Demo script to test CodeGrok MCP tools without an MCP client.
Exercises all tools and shows their behavior.

Usage:
  python scripts/demo_mcp.py [path_to_project]

If no path provided, creates a temporary sample project.
"""
import sys
import json
import asyncio
import tempfile
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Import MCP tools - access underlying functions from FastMCP wrappers
from codegrok_mcp.mcp.server import (
    learn as learn_tool,
    get_sources as get_sources_tool,
    get_stats as get_stats_tool,
    list_supported_languages as list_supported_languages_tool
)
from codegrok_mcp.mcp.state import reset_state

# Get underlying functions
learn = learn_tool.fn
get_sources = get_sources_tool.fn
get_stats = get_stats_tool.fn
list_supported_languages = list_supported_languages_tool.fn

console = Console()


def create_sample_project(base_path: Path) -> Path:
    """Create a minimal sample project for demo."""
    project = base_path / "sample_project"
    project.mkdir(parents=True, exist_ok=True)

    # main.py
    (project / "main.py").write_text('''"""Main application module."""

def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


def goodbye(name: str) -> str:
    """Say goodbye to someone."""
    return f"Goodbye, {name}!"


class Calculator:
    """A simple calculator for basic arithmetic operations."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def divide(self, a: int, b: int) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


if __name__ == "__main__":
    print(hello("World"))
    calc = Calculator()
    print(f"2 + 3 = {calc.add(2, 3)}")
''')

    # utils.py
    (project / "utils.py").write_text('''"""Utility functions for the application."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Dictionary containing configuration values.
    """
    with open(path) as f:
        return json.load(f)


def save_config(path: str, config: Dict[str, Any]) -> None:
    """Save configuration to a JSON file.

    Args:
        path: Path to save the configuration.
        config: Configuration dictionary to save.
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def find_files(directory: str, extension: str) -> list:
    """Find all files with a given extension in a directory.

    Args:
        directory: Directory to search in.
        extension: File extension to filter by (e.g., '.py').

    Returns:
        List of file paths matching the extension.
    """
    dir_path = Path(directory)
    return list(dir_path.glob(f"**/*{extension}"))


CONSTANT_VALUE = 42
DEFAULT_CONFIG = {"debug": False, "log_level": "INFO"}
''')

    # models.py
    (project / "models.py").write_text('''"""Data models for the application."""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class User:
    """Represents a user in the system."""
    id: int
    name: str
    email: str
    created_at: datetime
    is_active: bool = True

    def deactivate(self) -> None:
        """Deactivate the user account."""
        self.is_active = False

    def get_display_name(self) -> str:
        """Get the user's display name."""
        return f"{self.name} <{self.email}>"


@dataclass
class Project:
    """Represents a project with multiple users."""
    id: int
    name: str
    description: str
    owner: User
    members: List[User]
    created_at: datetime

    def add_member(self, user: User) -> None:
        """Add a new member to the project."""
        if user not in self.members:
            self.members.append(user)

    def remove_member(self, user: User) -> None:
        """Remove a member from the project."""
        if user in self.members:
            self.members.remove(user)

    def get_member_count(self) -> int:
        """Get the number of members in the project."""
        return len(self.members)
''')

    return project


def format_json(data: dict) -> str:
    """Format dict as colored JSON."""
    return json.dumps(data, indent=2, default=str)


def print_tool_call(name: str, params: dict = None):
    """Print a tool call header."""
    params_str = format_json(params) if params else "{}"
    console.print(f"\n[bold cyan]>>> Calling:[/bold cyan] [yellow]{name}[/yellow]")
    if params:
        console.print(Panel(
            Syntax(params_str, "json", theme="monokai"),
            title="Parameters",
            border_style="dim"
        ))


def print_result(result: dict):
    """Print a tool result."""
    console.print(Panel(
        Syntax(format_json(result), "json", theme="monokai"),
        title="Result",
        border_style="green"
    ))


def run_demo(project_path: Path):
    """Run the full demo sequence."""

    console.print(Panel.fit(
        "[bold]CodeGrok MCP Demo[/bold]\n"
        "Testing all MCP tools with real output",
        border_style="blue"
    ))

    console.print(f"\n[bold]Project path:[/bold] {project_path}\n")

    # 1. list_supported_languages
    console.rule("[bold magenta]1. list_supported_languages[/bold magenta]")
    console.print("Shows all programming languages and file extensions supported.")

    print_tool_call("list_supported_languages")
    result = list_supported_languages()

    # Pretty print as table
    table = Table(title="Supported Languages")
    table.add_column("Language", style="cyan")
    table.add_column("Extensions", style="green")
    for lang, exts in sorted(result["languages"].items()):
        table.add_row(lang, ", ".join(exts))
    console.print(table)

    # 2. learn (full index)
    console.rule("[bold magenta]2. learn (full index)[/bold magenta]")
    console.print("Indexes the codebase for semantic search.")

    print_tool_call("learn", {"path": str(project_path), "mode": "auto"})

    # learn is async, so we run it with asyncio
    result = asyncio.run(learn(path=str(project_path), mode="auto"))
    print_result(result)

    # 3. get_stats
    console.rule("[bold magenta]3. get_stats[/bold magenta]")
    console.print("Shows statistics about the indexed codebase.")

    print_tool_call("get_stats")
    result = get_stats()
    print_result(result)

    # 4. get_sources - multiple queries
    console.rule("[bold magenta]4. get_sources (semantic search)[/bold magenta]")
    console.print("Searches for relevant code using natural language queries.")

    queries = [
        "calculator add numbers",
        "load configuration from JSON file",
        "user data model",
    ]

    for query in queries:
        print_tool_call("get_sources", {"question": query, "n_results": 3})
        result = get_sources(question=query, n_results=3)

        # Show formatted sources nicely
        console.print(f"\n[bold green]Found {len(result.get('sources', []))} sources:[/bold green]")
        for i, source in enumerate(result.get("sources", [])[:3], 1):
            console.print(f"\n[dim]--- Source {i} ---[/dim]")
            console.print(source[:500] + "..." if len(source) > 500 else source)

    # 5. Demonstrate incremental reindex
    console.rule("[bold magenta]5. learn (incremental reindex)[/bold magenta]")
    console.print("Modifying a file and re-running learn to show incremental update.")

    # Modify a file
    new_function = '''

def new_feature() -> str:
    """A newly added feature for demo."""
    return "This is new!"
'''
    main_file = project_path / "main.py"
    original_content = main_file.read_text()
    main_file.write_text(original_content + new_function)
    console.print("[yellow]Added new_feature() to main.py[/yellow]")

    print_tool_call("learn", {"path": str(project_path), "mode": "auto"})
    result = asyncio.run(learn(path=str(project_path), mode="auto"))
    print_result(result)

    # 6. Search for the new function
    console.rule("[bold magenta]6. Search for newly added function[/bold magenta]")
    print_tool_call("get_sources", {"question": "new feature function", "n_results": 2})
    result = get_sources(question="new feature function", n_results=2)
    for source in result.get("sources", [])[:2]:
        console.print(source[:400] + "..." if len(source) > 400 else source)

    # 7. Demonstrate load (reload from disk)
    console.rule("[bold magenta]7. learn (load_only mode)[/bold magenta]")
    console.print("Clear state and reload existing index from disk.")

    reset_state()
    console.print("[yellow]State cleared[/yellow]")

    print_tool_call("learn", {"path": str(project_path), "mode": "load_only"})
    result = asyncio.run(learn(path=str(project_path), mode="load_only"))
    print_result(result)

    # Final stats
    console.rule("[bold magenta]8. Final stats[/bold magenta]")
    print_tool_call("get_stats")
    result = get_stats()
    print_result(result)

    # Summary
    console.print(Panel.fit(
        "[bold green]Demo Complete![/bold green]\n\n"
        "All 6 MCP tools exercised:\n"
        "  - list_supported_languages\n"
        "  - learn (full, incremental, load_only modes)\n"
        "  - get_stats\n"
        "  - get_sources\n\n"
        "The .codegrok/ directory was created in the project folder.",
        border_style="green"
    ))


def main():
    """Main entry point."""
    temp_dir = None

    try:
        if len(sys.argv) > 1:
            # Use provided path
            project_path = Path(sys.argv[1]).resolve()
            if not project_path.exists():
                console.print(f"[red]Error: Path does not exist: {project_path}[/red]")
                sys.exit(1)
            if not project_path.is_dir():
                console.print(f"[red]Error: Path is not a directory: {project_path}[/red]")
                sys.exit(1)
        else:
            # Create temp project
            temp_dir = tempfile.mkdtemp(prefix="codegrok_demo_")
            project_path = create_sample_project(Path(temp_dir))
            console.print(f"[dim]Created temporary sample project at: {project_path}[/dim]")

        run_demo(project_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise
    finally:
        # Clean up temp dir if we created one
        if temp_dir:
            cleanup = input("\nClean up temporary project? [Y/n]: ").strip().lower()
            if cleanup != 'n':
                shutil.rmtree(temp_dir)
                console.print("[dim]Cleaned up temporary files[/dim]")
            else:
                console.print(f"[dim]Temporary project kept at: {temp_dir}[/dim]")


if __name__ == "__main__":
    main()
