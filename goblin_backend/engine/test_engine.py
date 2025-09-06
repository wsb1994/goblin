
import pytest
from engine.engine import Engine, ExecutionStep

def test_plan_loads_correctly():
    # Example TOML string for a plan matching Engine's expected format
    example_toml = """
    [[steps]]
    name = "step1"
    function = "func1"
    inputs = ["default_input"]

    [[steps]]
    name = "step2"
    function = "func2"
    inputs = ["step1"]
    """
    engine = Engine()
    plan = engine.load_plan(example_toml)
    assert len(plan) == 2
    assert plan[0].name == "step1"
    assert plan[0].function == "func1"
    assert plan[0].inputs == ["default_input"]
    assert plan[1].name == "step2"
    assert plan[1].function == "func2"
    assert plan[1].inputs == ["step1"]
