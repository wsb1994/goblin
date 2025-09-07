
import toml
from typing import Dict, List, Any
import subprocess
import time
from threading import Lock
from engine.script import Script, Plan



import os

class Engine:
    def __init__(self):
        self.scripts: Dict[str, Script] = {}
        self.results_cache: Dict[str, Any] = {}
        self.scripts_lock = Lock()
        self.cache_lock = Lock()
        self.auto_discover_scripts()

    def auto_discover_scripts(self, scripts_dir: str = None):
        """Automatically discover scripts in /scripts/ subfolders by reading goblin.toml files."""
        if scripts_dir is None:
            scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        if not os.path.isdir(scripts_dir):
            return
        for entry in os.listdir(scripts_dir):
            subdir = os.path.join(scripts_dir, entry)
            if os.path.isdir(subdir):
                goblin_toml = os.path.join(subdir, "goblin.toml")
                if os.path.isfile(goblin_toml):
                    with open(goblin_toml, "r") as f:
                        config = toml.load(f)
                        # Support both [goblin] and flat TOML
                        script_data = config.get("goblin", config)
                        name = script_data.get("name", entry)
                        script_data = dict(script_data)
                        script_data["path"] = subdir
                        self.scripts[name] = Script(**script_data)

    def load_scripts(self, script_config: str) -> None:
        """Load script configurations from TOML string (manual override)."""
        config = toml.loads(script_config)
        with self.scripts_lock:
            for script_name, script_data in config.items():
                self.scripts[script_name] = Script(**script_data)

    def load_plan(self, plan_config_path: str) -> Plan:
        """Parse execution plan from TOML file path and return a single Plan with multiple steps"""
        if not os.path.isfile(plan_config_path):
            raise FileNotFoundError(f"Plan config file not found: {plan_config_path}")
            
        with open(plan_config_path, 'r') as f:
            config = toml.load(f)
        
        steps = config.get("steps", [])
        if not steps:
            raise ValueError("No steps found in plan config")
        
        # Find all script names in the plan
        script_names = set()
        for step in steps:
            if isinstance(step, dict):
                script_names.add(step.get('name'))
            else:
                script_names.add(step)

        # Load all required scripts from the scripts directory
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        print("====")
        print(scripts_dir)
        for script_name in script_names:
            script_dir = os.path.join(scripts_dir, script_name)
            if os.path.isdir(script_dir):
                goblin_toml = os.path.join(script_dir, "goblin.toml")
                if os.path.isfile(goblin_toml):
                    with open(goblin_toml, "r") as f:
                        script_config = toml.load(f)
                        print('~~~')
                        print(script_config)
                        script_data = script_config.get("goblin", script_config)
                        print('```')
                        print(script_data)
                        script_data = dict(script_data)
                        script_data["path"] = script_dir
                        with self.scripts_lock:
                            self.scripts[script_name] = Script(**script_data)
        print('----')
        print(config)

        
        return Plan(name=config['name'], steps=steps)



    def run_script(self, script: Script, inputs: List[Any]) -> Any:
        """Execute a script with given inputs"""
        with self.scripts_lock:
            if script.name not in self.scripts:
                raise ValueError(f"Script {script.name} not found")
            script = self.scripts[script.name]

        if script.require_test:
            test_result = self._run_test(script)
            if not test_result:
                raise RuntimeError(f"Test failed for script {script.name}")

        try:
            # Save current directory
            original_dir = os.getcwd()
            
            # Change to script directory if specified
            if script.path:
                os.chdir(script.path)
            
            # Run the process with inputs (properly escape JSON strings)
            import shlex
            escaped_inputs = [shlex.quote(str(i)) for i in inputs]
            command = f"{script.command} {' '.join(escaped_inputs)}"
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > script.timeout:
                    process.kill()
                    raise TimeoutError(f"Script {script.name} timed out")
                time.sleep(0.1)

            output, error = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Script {script.name} failed: {error.decode()}")

            return output.decode().strip()

        except Exception as e:
            raise RuntimeError(f"Error running script {script.name}: {str(e)}")
        finally:
            # Always return to original directory
            if 'original_dir' in locals():
                os.chdir(original_dir)

    def _run_test(self, script: Script) -> bool:
        """Run test command for a script"""
        if not script.test_command:
            return True
            
        process = subprocess.run(
            script.test_command,
            shell=True,
            capture_output=True,
            text=True
        )
        return process.returncode == 0 and process.stdout.strip().lower() == 'true'


    def execute_plan(self, plan: Plan, default_input: Any = None) -> Dict[str, Any]:
        """Execute a plan of steps with optional default input"""
        # Initialize cache with default input if provided
        with self.cache_lock:
            if default_input is not None:
                self.results_cache['default_input'] = default_input

        for step in plan.steps:
            with self.cache_lock:
                if isinstance(step, dict):
                    step_name = step.get('name')
                    inputs = step.get('inputs', [])
                else:
                    step_name = step
                    inputs = []
                
                # Process inputs with format string replacement
                input_values = []
                for input_str in inputs:
                    try:
                        # Check if input_str is a reference to a cached result
                        if input_str in self.results_cache:
                            # Use the cached result directly
                            input_values.append(str(self.results_cache[input_str]))
                        else:
                            # Try to format the input string with the cache values
                            formatted_input = input_str.format(**self.results_cache)
                            input_values.append(formatted_input)
                    except KeyError:
                        # If formatting fails, use the original input string
                        input_values.append(input_str)
            
            script_instance = None
            if step_name in self.scripts:
                script_instance = self.scripts[step_name]
            else:
                raise ValueError(f"Script {step_name} not found in loaded scripts.")
            
            # Debug print to show the command that will be executed
            print(f"Executing: {script_instance.command} {' '.join(input_values)}")
            
            result = self.run_script(script_instance, input_values)
            print("console log, result from script: ", result)
            with self.cache_lock:
                self.results_cache[step_name] = result

        return dict(self.results_cache)  # Return a copy to be thread-safe
