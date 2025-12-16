import os
import subprocess
import json
from pathlib import Path
from shutil import which

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    """Load config file from package directory."""
    config_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json file not found at {config_path}. Please create this file with the required configuration.")
    with open(config_path, "r") as f:
        return json.load(f)

class RunJulia:
    """Orchestrates Julia script execution using the existing functions in this module.
    """
    
    def __init__(self, output_dir=None, julia_env_path=None):
        """Create a Julia runner.
        
        Args:
            output_dir: Optional output directory. If None, uses output_dir from config.json.
            julia_env_path: Optional Julia environment path. If None, uses julia_env_path from config.json.
        """
        # Load config to get paths
        config = load_config()
        
        # Julia scripts are always in the package directory
        self.julia_scripts_dir = os.path.join(BASE_DIR, "julia")
        
        # Output directory from parameter or config
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = config.get("path", BASE_DIR)
            
        # Julia environment path from parameter or config
        if julia_env_path is not None:
            self.julia_env_path = julia_env_path
        else:
            self.julia_env_path = config.get("julia_env_path")
            if not self.julia_env_path:
                raise ValueError("julia_env_path not found in config.json")
        
        # Check if the environment exists
        if not os.path.exists(self.julia_env_path):
            raise RuntimeError(f"Julia environment not found at {self.julia_env_path}")
        
        # Use standard julia command
        self.julia_cmd = ["julia"]
        
        print(f"Using Julia environment: {self.julia_env_path}")
        print(f"Using output directory: {self.output_dir}")
    
    def CO(self):
        """Run the CO.jl Julia script."""
        subprocess.run([
            *self.julia_cmd,
            f"--project={self.julia_env_path}",
            os.path.join(self.julia_scripts_dir, "CO.jl")
        ], check=True, text=True, cwd=self.output_dir)
        
    def SynthPop(self):
        """Run the synthpop.jl Julia script."""
        print("Running synthpop.jl...")
        try:
            result = subprocess.run([
                *self.julia_cmd,
                f"--project={self.julia_env_path}",
                os.path.join(self.julia_scripts_dir, "synthpop.jl")
            ], check=True, text=True, cwd=self.output_dir, capture_output=True)
            print("synthpop.jl completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            print(f"synthpop.jl failed with exit code {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise
        
    def Export(self):
        """Run the export_synthpop.jl and the export_network.jl Julia scripts."""
        subprocess.run([
            *self.julia_cmd,
            f"--project={self.julia_env_path}",
            os.path.join(self.julia_scripts_dir, "export_synthpop.jl")
        ], check=True, text=True, cwd=self.output_dir)
        subprocess.run([
            *self.julia_cmd,
            f"--project={self.julia_env_path}",
            os.path.join(self.julia_scripts_dir, "export_network.jl")
        ], check=True, text=True, cwd=self.output_dir)
    
    def run_all(self):
        """Run all Julia scripts in sequence."""
        self.CO()
        self.SynthPop()
        self.Export()

def main():
    runner = RunJulia()
    # runner.run_all()

if __name__ == "__main__":
    main()

