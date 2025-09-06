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

        print('~-~-~-~_~-~-~')
        myresult = engine.scripts.get("Example")
        print(myresult)
        
        engine.execute_plan(plan, '{"Hello":"world"}')
    finally:
        os.chdir(old_cwd)



def test_engine_example_hate_speech_plan(tmp_path):
    # set your env variable for ANTHROPIC_API_KEY before running this test
    # Change working directory to temp dir for subprocess
    import os
    old_cwd = os.getcwd()
    try:
        engine = Engine()
        engine.auto_discover_scripts("/home/wsb/Documents/Binaries/goblin/goblin_backend/scripts")
        plan = engine.load_plan("/home/wsb/Documents/Binaries/goblin/goblin_backend/plans/AB.toml")
        print("===plan===")
        print(plan)

        # print('~-~-~-~_~-~-~')
        # myresult = engine.scripts.get("Example")
        # print(myresult)
        
        engine.execute_plan(plan, '{"id": "sample_001", "timestamp": "2025-01-06T17:11:00Z", "comment": "I really enjoyed the community event last weekend. It was great to see people from different backgrounds coming together to help clean up the local park. The organizers did an excellent job coordinating everything. Except the fact that they are black lol", "user_id": "user_12345", "platform": "community_forum", "category": "local_events", "metadata": {"location": "Springfield Park", "event_type": "community_service", "participants": 45}}')
    finally:
        os.chdir(old_cwd)
