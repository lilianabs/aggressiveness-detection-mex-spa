import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate_model import evaluate, save_predictions
from src.utils import log_model_to_wandb


def split_dataset(X, y, test_size: float, random_state: int, stratify: bool = True):
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, ngram_range: tuple = (1, 2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def cross_validate(model, X, y, cv_folds: int = 5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model.fit(X_train_cv, y_train_cv)
        metrics, _, _ = evaluate(model, X_val_cv, y_val_cv)

        scores["accuracy"].append(metrics["accuracy"])
        scores["precision"].append(metrics["precision"])
        scores["recall"].append(metrics["recall"])
        scores["f1"].append(metrics["f1"])

    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in scores.items()}


def train(model, X_train, y_train):
    model.fit(X_train, y_train)

    return model


def run_train(
    model,
    model_name: str,
    config: Dict[str, Any],
    df: Any,
    df_raw: Any,
    predictions_path: Path,
    n_gram_range: Tuple[int, int],
    project_name: str,
) -> None:
    text_column = config["task"]["text_column"]
    label_column = config["task"]["label_column"]
    test_size = config["reproducibility"]["train_test_split"]["test_size"]
    random_state = config["reproducibility"]["train_test_split"]["random_state"]
    stratify = config["reproducibility"]["train_test_split"]["stratify"]

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
    log_model_to_wandb(
        project_name, model_name, cv_scores, preprocess_steps, n_gram_range
    )
