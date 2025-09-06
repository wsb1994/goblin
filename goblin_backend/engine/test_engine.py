import unittest
import toml
from goblin_backend.engine.engine import load_plan  # Assuming you have a function to load plans

class TestEnginePlanLoad(unittest.TestCase):
    def setUp(self):
        # Example TOML string for a plan
        self.example_toml = """
        [plan]
        name = "TestPlan"
        steps = ["step1", "step2"]
        """

    def test_plan_loads_correctly(self):
        # Parse TOML string
        plan_data = toml.loads(self.example_toml)
        # Load plan using engine logic
        plan = load_plan(plan_data)
        # Assertions (customize as per your plan structure)
        self.assertEqual(plan.name, "TestPlan")
        self.assertEqual(plan.steps, ["step1", "step2"])

if __name__ == "__main__":
    unittest.main()
