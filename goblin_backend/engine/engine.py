import toml
from typing import Dict, List, Any
import subprocess
import time
from dataclasses import dataclass
from threading import Lock

@dataclass
class GoblinScript:
    name: str
    command: str
    timeout: int = 500
    test_command: str = ""
    require_test: bool = False

@dataclass
class ExecutionStep:
    name: str
    function: str
    inputs: List[str]

class Engine:
    def __init__(self):
        self.scripts: Dict[str, GoblinScript] = {}
        self.results_cache: Dict[str, Any] = {}
        self.scripts_lock = Lock()
        self.cache_lock = Lock()
        
    def load_scripts(self, script_config: str) -> None:
        """Load script configurations from TOML string"""
        config = toml.loads(script_config)
        with self.scripts_lock:
            for script_name, script_data in config.items():
                self.scripts[script_name] = GoblinScript(**script_data)

    def load_plan(self, plan_config: str) -> List[ExecutionStep]:
        """Parse execution plan from TOML string"""
        config = toml.loads(plan_config)
        steps = []
        for step in config.get("steps", []):
            steps.append(ExecutionStep(**step))
        return steps

    def run_script(self, script_name: str, inputs: List[Any]) -> Any:
        """Execute a script with given inputs"""
        with self.scripts_lock:
            if script_name not in self.scripts:
                raise ValueError(f"Script {script_name} not found")
            script = self.scripts[script_name]
        
        if script.require_test:
            test_result = self._run_test(script)
            if not test_result:
                raise RuntimeError(f"Test failed for script {script_name}")

        try:
            process = subprocess.Popen(
                script.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > script.timeout:
                    process.kill()
                    raise TimeoutError(f"Script {script_name} timed out")
                time.sleep(0.1)

            output, error = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Script {script_name} failed: {error.decode()}")
            
            return output.decode().strip()

        except Exception as e:
            raise RuntimeError(f"Error running script {script_name}: {str(e)}")

    def _run_test(self, script: GoblinScript) -> bool:
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

    def execute_plan(self, plan: List[ExecutionStep]) -> Dict[str, Any]:
        """Execute a plan of steps"""
        for step in plan:
            with self.cache_lock:
                input_values = [
                    self.results_cache[input_name] 
                    if input_name in self.results_cache 
                    else input_name 
                    for input_name in step.inputs
                ]
            
            result = self.run_script(step.function, input_values)
            
            with self.cache_lock:
                self.results_cache[step.name] = result
            
        return dict(self.results_cache)  # Return a copy to be thread-safe