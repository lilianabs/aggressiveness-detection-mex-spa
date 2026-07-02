import os
from typing import Any

import wandb
from sklearn.svm import SVC
from sklearn.base import clone
from src.utils import load_config, get_config_path, get_project_root, load_csv
from src.train_model import split_dataset, vectorize_text, cross_validate
from src.evaluate import evaluate, save_predictions
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)


def main():
    n_gram_range = (1, 4)
    params_dict: dict[str, Any] = {
        "class_weight": "balanced",
        "kernel": "linear"
    }
    params_str = "_".join(
        f"{k}_{v}" for k, v in params_dict.items() if k != "random_state"
    )
    model_name = f"SVM_ngram_{n_gram_range[0]}-{n_gram_range[1]}" + (
        f"_{params_str}" if params_str else ""
    )

    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    local_raw_data_path = (
        project_root / config["data"]["local_raw_data_path"]
    ).resolve()
    local_preprocessed_data_path = (
        project_root / config["data"]["local_preprocessed_data_path"]
    ).resolve()
    predictions_path = (
        project_root / config["data"]["local_predictions_file"].format(model_name=model_name)
    ).resolve()
    text_column = config["task"]["text_column"]
    label_column = config["task"]["label_column"]

    test_size = config["reproducibility"]["train_test_split"]["test_size"]
    random_state = config["reproducibility"]["train_test_split"]["random_state"]
    stratify = config["reproducibility"]["train_test_split"]["stratify"]

    params_dict["random_state"] = random_state

    print(f"Loading preprocessed data from {local_preprocessed_data_path}...")
    df = load_csv(str(local_preprocessed_data_path))
    # We load the raw data to get the original text for saving predictions, but we don't use it for training or evaluation
    df_raw = load_csv(str(local_raw_data_path))
    print(df.head())
    print(f"Loaded {len(df)} rows\n")

    X = df[text_column].values
    X_raw = df_raw[text_column].values
    y = df[label_column].values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    _, X_test_raw, _, _ = split_dataset(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}\n")

    print("Vectorizing text...")
    X_train_tfidf, X_test_tfidf = vectorize_text(
        X_train, X_test, ngram_range=n_gram_range
    )
    print(f"TF-IDF shape: {X_train_tfidf.shape}\n")

    print("Creating SVM model...")
    model = SVC(**params_dict)
    print(f"Model: {model}\n")

    print("Performing cross-validation...")
    cv_scores = cross_validate(clone(model), X_train_tfidf, y_train, cv_folds=5)
    print("Cross-validation scores:")
    for metric, values in cv_scores.items():
        print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print()

    print("Training final model on full train set...")
    model.fit(X_train_tfidf, y_train)
    print("Model trained!\n")

    print("Evaluating on test set...")
    metrics, cm, report = evaluate(model, X_test_tfidf, y_test)
    print("Test set scores:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
    print()

    print("Confusion Matrix:")
    print(cm)
    print()

    print("Classification Report:")
    print(report)
    print()

    print("Saving model predictions...")
    y_pred = model.predict(X_test_tfidf)
    save_predictions(X_test, X_test_raw, y_test, y_pred, predictions_path)
    print(f"Predictions saved to {predictions_path}\n")

    print("Logging to Weights & Biases...")
    preprocess_steps = config["task"]["preprocessing_steps"]
    wandb.init(
        project="aggressiveness-detection",
        name=model_name,
        config={
            "model": f"{model_name}",
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

    print("Done!")


if __name__ == "__main__":
    main()
