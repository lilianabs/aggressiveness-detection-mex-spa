"""
Self-contained experiment: SVM trained on combined word-level + character-level
TF-IDF features via FeatureUnion.

This script intentionally does NOT call src.train_model.run_train(), since that
function only supports a single TfidfVectorizer. Equivalent orchestration logic
(split -> vectorize -> cross-validate -> fit -> evaluate -> save -> log) is
reimplemented here inline, reusing all other src/ utilities unchanged.
"""

import os
from typing import Any

import wandb
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from dotenv import load_dotenv

from src.utils import (
    load_config,
    get_config_path,
    get_project_root,
    load_training_data,
    get_predictions_path,
    log_model_to_wandb,
)
from src.train_model import split_dataset, cross_validate
from src.evaluate_model import evaluate, save_predictions

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)


def build_combined_vectorizer(
    word_ngram_range: tuple, char_ngram_range: tuple
) -> FeatureUnion:
    """Combine word-level and char-level TF-IDF vectorizers into one feature space."""
    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(analyzer="word", ngram_range=word_ngram_range),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range),
        ),
    ]
    return FeatureUnion(transformers)  # type: ignore


def main():
    word_ngram_range = (1, 4)
    char_ngram_range = (2, 5)

    params_dict: dict[str, Any] = {"class_weight": "balanced", "kernel": "linear"}
    params_str = "_".join(
        f"{k}_{v}" for k, v in params_dict.items() if k != "random_state"
    )
    model_name = (
        f"SVM_combined_word_{word_ngram_range[0]}-{word_ngram_range[1]}"
        f"_char_wb_{char_ngram_range[0]}-{char_ngram_range[1]}"
        + (f"_{params_str}" if params_str else "")
    )
    print(f"Model name: {model_name}")

    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    df, df_raw, text_column, label_column = load_training_data(config, project_root)
    predictions_path = get_predictions_path(config, project_root, model_name)
    project_name = config["project"]["name"]

    test_size = config["reproducibility"]["train_test_split"]["test_size"]
    random_state = config["reproducibility"]["train_test_split"]["random_state"]
    stratify = config["reproducibility"]["train_test_split"]["stratify"]
    params_dict["random_state"] = random_state

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

    print("Building combined word + char_wb TF-IDF FeatureUnion...")
    combined_vectorizer = build_combined_vectorizer(word_ngram_range, char_ngram_range)
    print("Vectorizing text (word TF-IDF + char_wb TF-IDF)...")
    X_train_combined = combined_vectorizer.fit_transform(X_train)
    X_test_combined = combined_vectorizer.transform(X_test)
    print(f"Combined feature shape: {X_train_combined.shape}\n")

    print("Creating SVM model...")
    model = SVC(**params_dict)
    print(f"Model: {model}\n")

    print("Performing cross-validation...")
    cv_scores = cross_validate(clone(model), X_train_combined, y_train, cv_folds=5)
    print("Cross-validation scores:")
    for metric, values in cv_scores.items():
        print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print()

    print("Training final model on full train set...")
    model.fit(X_train_combined, y_train)
    print("Model trained!\n")

    print("Evaluating on test set...")
    metrics, cm, report = evaluate(model, X_test_combined, y_test)
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
    y_pred = model.predict(X_test_combined)
    save_predictions(X_test, X_test_raw, y_test, y_pred, predictions_path)
    print(f"Predictions saved to {predictions_path}\n")

    print("Logging to Weights & Biases...")
    preprocess_steps = config["task"]["preprocessing_steps"]
    # log_model_to_wandb's `n_gram_range` param is typed Tuple[int, int] but is only
    # ever interpolated into an f-string, so it's safe to pass a descriptive string
    # here instead of a single tuple -- there are two distinct n-gram ranges (word
    # and char_wb) in this experiment and neither alone is representative.
    n_gram_info = f"word{word_ngram_range}+char_wb{char_ngram_range}"
    log_model_to_wandb(  # type: ignore
        project_name, model_name, cv_scores, preprocess_steps, n_gram_info
    )

    print("Done!")


if __name__ == "__main__":
    main()
