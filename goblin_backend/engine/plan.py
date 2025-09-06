import os
import toml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class GoblinScript:
    name: str
    command: str
    timeout: int
    test_command: Optional[str]
    require_test: bool
    script_path: Path

def find_project_root() -> Path:
    """Navigate up until we find a directory containing 'goblin_backend'"""
    current = Path.cwd()
    while current.name != "goblin_backend" and current.parent != current:
        current = current.parent
    return current.parent

def load_scripts() -> List[GoblinScript]:
    """Load all scripts from the scripts directory"""
    project_root = find_project_root()
    scripts_dir = project_root / "scripts"
    scripts = []

    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found at {scripts_dir}")

    for script_dir in scripts_dir.iterdir():
        if not script_dir.is_dir():
            continue

        toml_path = script_dir / "goblin.toml"
        if not toml_path.exists():
            continue

        with open(toml_path, "rb") as f:
            try:
                config = toml.load(f)
                script = GoblinScript(
                    name=config.get("name", ""),
                    command=config.get("command", ""),
                    timeout=config.get("timeout", 500),
                    test_command=config.get("test_command"),
                    require_test=config.get("require_test", False),
                    script_path=script_dir
                )
                scripts.append(script)
            except Exception as e:
                print(f"Error loading {toml_path}: {e}")

    return scripts

