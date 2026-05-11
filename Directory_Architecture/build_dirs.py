import platform as plat #used for detecting which OS we are currently on. 
import json # used for reading Json structure files.
from pathlib import Path #Used for creating path objects and making directories. 
import argparse # used for CLI flags 
import sys # provides clean error texts. 
import yaml #used for reading structure yaml 

#-----------------------------
# System Detection
#-----------------------------
ROOT = {
    "Linux": "/home/tdeibert",
    "Windows": r"C:\Users\cowboy\OneDrive - UAB - The University of Alabama at Birmingham"
}

def get_system() -> str:
    system = plat.system()
    if system == 'Windows':
        return 'Windows'
    elif system == 'Linux':
        return 'Linux'
    else:
        sys.exit(f'Unsupported System: {system}')

def get_root() -> Path: 
    system = get_system()
    system_root = Path(ROOT[system])
    return system_root

def load_config(config_path: Path) -> dict:
    suffix = config_path.suffix 
    if suffix == ".yaml":
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        sys.exit(f"Unsupported Config Format: {suffix}")

def build_dirs(structure:dict, current_path, dry_run: bool = False):
    for key, value in structure.items():
        new_path = current_path/ key
        if dry_run:
            print(new_path)
        else:
            new_path.mkdir(parents = True, exist_ok = True)
        if isinstance(value, dict):
            build_dirs(value, new_path, dry_run)
        else:
            if value is None:
                pass 
    
#----------------------------------------
# Main Argument Parse 
#----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Creates Stable Directories")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the config file(structure.yaml, structure.json, or struture.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check Directory Structure from(structure.ymal, structure.json, or structure.py)"
    )
    args = parser.parse_args()
    build_root = get_root()
    structure = load_config(args.config)
    print(f"New Directories Will be created under {build_root}")
    answer = input("proceed?: [y/n]")
    if answer.lower() != "y":
        sys.exit("aborted")
    build_dirs(structure,build_root,args.dry_run)

if __name__ == "__main__":
    main()




