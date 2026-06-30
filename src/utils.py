import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_config_path() -> Path:
    return get_project_root() / "config.yaml"


def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading {file_path}: {e}")
