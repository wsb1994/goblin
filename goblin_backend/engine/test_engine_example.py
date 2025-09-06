import json
from engine.engine import Engine
from engine.script import Plan


def test_engine_example_plan(tmp_path):

    # Change working directory to temp dir for subprocess
    import os
    old_cwd = os.getcwd()
    try:
        engine = Engine()
        plan = engine.load_plan("/home/wsb/Documents/Binaries/goblin/goblin_backend/plans/example.toml")
        print("===plan===")
        print(plan)
        #result = engine.execute_plan(plan)
        # The output should be the JSON string from the script
      

        print('~-~-~-~_~-~-~')
        myresult = engine.scripts.get("Example")
        print(myresult)
        
        engine.execute_plan(plan)
    finally:
        os.chdir(old_cwd)
