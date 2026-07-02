import yaml
import pandas as pd
import wandb
from pathlib import Path
from typing import Any, Dict, Tuple


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


def load_training_data(
    config: dict, project_root: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    local_raw_data_path = (
        project_root / config["data"]["local_raw_data_path"]
    ).resolve()
    local_preprocessed_data_path = (
        project_root / config["data"]["local_preprocessed_data_path"]
    ).resolve()

    print(f"Loading preprocessed data from {local_preprocessed_data_path}...")
    df = load_csv(str(local_preprocessed_data_path))
    df_raw = load_csv(str(local_raw_data_path))
    print(df.head())
    print(f"Loaded {len(df)} rows\n")

    text_column = config["task"]["text_column"]
    label_column = config["task"]["label_column"]

    return df, df_raw, text_column, label_column


def get_predictions_path(config: dict, project_root: Path, model_name: str) -> Path:
    return (
        project_root
        / config["data"]["local_predictions_file"].format(model_name=model_name)
    ).resolve()


def log_model_to_wandb(
    project_name: str,
    model_name: str,
    cv_scores: Dict[str, Any],
    preprocess_steps: list,
    n_gram_range: Tuple[int, int],
) -> None:
    wandb.init(
        project=project_name,
        name=model_name,
        config={
            "model": model_name,
            "Preprocessing": preprocess_steps,
            "TF-IDF": f"TF-IDF vectorization with n-grams {n_gram_range}",
        },
    )
    wandb.log({"cv_accuracy_mean": cv_scores["accuracy"]["mean"]})
    wandb.log({"cv_accuracy_std": cv_scores["accuracy"]["std"]})
    wandb.log({"cv_precision_mean": cv_scores["precision"]["mean"]})
    wandb.log({"cv_precision_std": cv_scores["precision"]["std"]})
    wandb.log({"cv_recall_mean": cv_scores["recall"]["mean"]})
    wandb.log({"cv_recall_std": cv_scores["recall"]["std"]})
    wandb.log({"cv_f1_mean": cv_scores["f1"]["mean"]})
    wandb.log({"cv_f1_std": cv_scores["f1"]["std"]})
    wandb.finish()
