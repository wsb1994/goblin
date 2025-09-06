from dataclasses import dataclass
from typing import List

@dataclass
class Script:
    name: str
    command: str
    timeout: int = 500
    test_command: str = ""
    require_test: bool = False
    path: str = ""

@dataclass
class Plan:
    name: str
    steps: List[str] = None
