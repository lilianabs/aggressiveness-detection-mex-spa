import os
from typing import Any

import wandb
from sklearn.svm import SVC
from src.utils import (
    load_config,
    get_config_path,
    get_project_root,
    load_training_data,
    get_predictions_path,
)
from src.train_model import run_train
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)


def main():
    n_gram_range = (1, 4)
    analyzer = "word"  # Use character-level analysis for SVM
    params_dict: dict[str, Any] = {"class_weight": "balanced", "kernel": "linear"}
    params_str = "_".join(
        f"{k}_{v}" for k, v in params_dict.items() if k != "random_state"
    )
    model_name = f"SVM_ngram_{n_gram_range[0]}-{n_gram_range[1]}"+ f"_analyzer_{analyzer}" + (
        f"_{params_str}" if params_str else ""
    )
    print(f"Model name: {model_name}")
    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    df, df_raw, text_column, label_column = load_training_data(config, project_root)
    predictions_path = get_predictions_path(config, project_root, model_name)
    project_name = config["project"]["name"]

    random_state = config["reproducibility"]["train_test_split"]["random_state"]
    params_dict["random_state"] = random_state

    print("Creating SVM model...")
    model = SVC(**params_dict)
    print(f"Model: {model}\n")

    run_train(
        model,
        model_name,
        config,
        df,
        df_raw,
        predictions_path,
        n_gram_range,
        project_name,
        analyzer
    )

    print("Done!")


if __name__ == "__main__":
    main()
