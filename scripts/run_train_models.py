import os
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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

n_gram_range = (1, 3)

MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "NaiveBayes": MultinomialNB(),
}


def main():
    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    project_name = config["project"]["name"]
    df, df_raw, text_column, label_column = load_training_data(config, project_root)

    preprocess_steps = config["task"]["preprocessing_steps"]

    for model_class_name, model in MODELS.items():
        model_name = f"{model_class_name}_ngram_{n_gram_range[0]}-{n_gram_range[1]}"
        predictions_path = get_predictions_path(config, project_root, model_name)

        print(f"\n{'=' * 50}")
        print(f"Training {model_class_name}...")
        print(f"{'=' * 50}\n")

        run_train(
            model,
            model_name,
            config,
            df,
            df_raw,
            predictions_path,
            n_gram_range,
            project_name,
        )

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
