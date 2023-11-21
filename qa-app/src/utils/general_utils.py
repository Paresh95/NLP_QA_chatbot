import yaml
from typing import Dict


def read_yaml_config(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f.read())
    except yaml.YAMLError as e:
        print(f"Error parsing parameters.yaml: {e}")
    except Exception as e:
        print(f"Unexpected error reading parameters.yaml: {e}")
    return {}
