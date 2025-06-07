"""
Tools Setup - Automatic installation and configuration of external tools

This script handles the installation and setup of all external tools and
dependencies in a unified way, hiding complexity from the user.
"""
import os
import sys
import argparse
import logging
import subprocess
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("owl.tools_setup")

# Define tools and their configurations
TOOLS = {
    "deplot": {
        "name": "DePlot",
        "description": "Extract data from charts and plots",
        "repo": "https://github.com/google-research/google-research.git",
        "path": "deplot",
        "requirements": [
            "pix2struct",
            "openai~=0.26.4",
            "scipy~=1.10.0"
        ],
        "post_install": """
            # Clone Pix2Struct repository
            git clone https://github.com/google-research/pix2struct.git
            cd pix2struct
            pip install -e .
            cd ..
        """
    },
    "nam": {
        "name": "Neural Additive Models",
        "description": "Interpretable predictions with transparent feature contributions",
        "repo": "https://github.com/google-research/google-research.git",
        "path": "neural_additive_models",
        "requirements": [
            "tensorflow>=2.4.0",
            "tensorflow-addons>=0.13.0",
            "numpy>=1.19.0",
            "pandas>=1.1.0",
            "scikit-learn>=0.24.0",
            "matplotlib>=3.3.0"
        ],
        "post_install": None
    },
    "dvrl": {
        "name": "Data Valuation using Reinforcement Learning",
        "description": "Assess the value of data points and documents",
        "repo": "https://github.com/google-research/google-research.git",
        "path": "dvrl",
        "requirements": [
            "tensorflow>=2.4.0",
            "tensorflow-probability>=0.12.0",
            "numpy>=1.19.0",
            "pandas>=1.1.0",
            "scikit-learn>=0.24.0"
        ],
        "post_install": None
    },
    "optlist": {
        "name": "Optimized Hyperparameter Lists",
        "description": "Optimized hyperparameters for machine learning models",
        "repo": "https://github.com/google-research/google-research.git",
        "path": "opt_list",
        "requirements": [
            "tensorflow>=2.4.0",
            "numpy>=1.19.0"
        ],
        "post_install": None
    }
}

class ToolsSetup:
    """Setup and manage external tools"""
    
    def __init__(self, tools_dir="/app/tools", verbose=False):
        """
        Initialize the tools setup
        
        Args:
            tools_dir: Directory to install tools
            verbose: Enable verbose output
        """
        self.tools_dir = Path(tools_dir)
        self.verbose = verbose
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Python path to include tools
        sys.path.append(str(self.tools_dir))
        os.environ["PYTHONPATH"] = f"{str(self.tools_dir)}:{os.environ.get('PYTHONPATH', '')}"
    
    def install_tool(self, tool_key):
        """
        Install a specific tool
        
        Args:
            tool_key: Key of the tool to install
            
        Returns:
            True if successful, False otherwise
        """
        if tool_key not in TOOLS:
            logger.error(f"Unknown tool: {tool_key}")
            return False
        
        tool = TOOLS[tool_key]
        logger.info(f"Installing {tool['name']}...")
        
        try:
            # Create tool directory
            tool_dir = self.tools_dir / tool_key
            tool_dir.mkdir(exist_ok=True)
            
            # Clone repository if it doesn't exist
            repo_dir = self.tools_dir / f"google-research-{tool_key}"
            if not repo_dir.exists():
                logger.info(f"Cloning repository for {tool['name']}...")
                cmd = f"git clone {tool['repo']} {repo_dir}"
                self._run_command(cmd)
            
            # Create symbolic link to tool directory
            source_dir = repo_dir / tool['path']
            if source_dir.exists() and not (tool_dir / tool['path']).exists():
                os.symlink(source_dir, tool_dir / tool['path'])
            
            # Install requirements
            logger.info(f"Installing requirements for {tool['name']}...")
            for req in tool['requirements']:
                cmd = f"pip install --no-cache-dir {req}"
                self._run_command(cmd)
            
            # Run post-install commands if any
            if tool['post_install']:
                logger.info(f"Running post-install steps for {tool['name']}...")
                for cmd in tool['post_install'].strip().split('\n'):
                    cmd = cmd.strip()
                    if cmd and not cmd.startswith('#'):
                        self._run_command(cmd, cwd=tool_dir)
            
            logger.info(f"Successfully installed {tool['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing {tool['name']}: {e}")
            return False
    
    def install_all_tools(self):
        """
        Install all available tools
        
        Returns:
            Number of successfully installed tools
        """
        logger.info("Installing all tools...")
        success_count = 0
        
        for tool_key in TOOLS:
            if self.install_tool(tool_key):
                success_count += 1
        
        logger.info(f"Installed {success_count}/{len(TOOLS)} tools successfully")
        return success_count
    
    def check_tool_availability(self, tool_key):
        """
        Check if a tool is available
        
        Args:
            tool_key: Key of the tool to check
            
        Returns:
            True if available, False otherwise
        """
        if tool_key not in TOOLS:
            return False
        
        tool = TOOLS[tool_key]
        tool_dir = self.tools_dir / tool_key / tool['path']
        return tool_dir.exists()
    
    def get_available_tools(self):
        """
        Get a list of available tools
        
        Returns:
            List of available tool keys
        """
        available_tools = []
        for tool_key in TOOLS:
            if self.check_tool_availability(tool_key):
                available_tools.append(tool_key)
        return available_tools
    
    def _run_command(self, cmd, cwd=None):
        """
        Run a shell command
        
        Args:
            cmd: Command to run
            cwd: Working directory
            
        Returns:
            Command output
        """
        if cwd is None:
            cwd = self.tools_dir
        
        if self.verbose:
            logger.info(f"Running: {cmd}")
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.PIPE if not self.verbose else None,
            text=True
        )
        
        if result.returncode != 0 and not self.verbose:
            logger.error(f"Command failed: {cmd}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            raise Exception(f"Command failed with exit code {result.returncode}")
        
        return result.stdout if result.stdout else ""

# Function to get available tools
def get_available_tools(tools_dir="/app/tools"):
    """Get list of available tools"""
    setup = ToolsSetup(tools_dir)
    return setup.get_available_tools()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup external tools")
    parser.add_argument("--install-all", action="store_true", help="Install all tools")
    parser.add_argument("--install", type=str, help="Install specific tool")
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--tools-dir", type=str, default="/app/tools", help="Directory to install tools")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    setup = ToolsSetup(args.tools_dir, args.verbose)
    
    if args.list:
        available = setup.get_available_tools()
        print(f"Available tools ({len(available)}/{len(TOOLS)}):")
        for tool_key in TOOLS:
            status = "✅ Installed" if tool_key in available else "❌ Not installed"
            print(f"  {TOOLS[tool_key]['name']} ({tool_key}): {status}")
            print(f"    {TOOLS[tool_key]['description']}")
            print()
    
    elif args.install:
        setup.install_tool(args.install)
    
    elif args.install_all:
        setup.install_all_tools()
    
    else:
        parser.print_help()