"""
Self-contained experiment: GridSearchCV hyperparameter tuning for the best model.

Best model baseline (per CLAUDE.md): SVM with char n-grams (1,4), kernel='linear',
class_weight='balanced', achieving CV F1 0.691 ± 0.017.

This script tunes SVM's C parameter together with TF-IDF char n-gram range using
GridSearchCV with F1 as the optimization metric. The vectorizer is embedded in a
sklearn Pipeline so GridSearchCV can refit TF-IDF per fold, avoiding train-test leakage.

Intentionally does NOT call src.train_model.run_train() — orchestration is reimplemented
inline, reusing src.utils, src.train_model.split_dataset only, and src.evaluate_model.
"""

import os
from typing import Any, Dict

import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from dotenv import load_dotenv

from src.utils import (
    load_config,
    get_config_path,
    get_project_root,
    load_training_data,
    get_predictions_path,
)
from src.train_model import split_dataset
from src.evaluate_model import evaluate, save_predictions

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)


def extract_cv_scores_for_best_params(
    cv_results: Dict[str, Any], best_index: int
) -> Dict[str, Dict[str, float]]:
    """
    Extract mean and std for all metrics from GridSearchCV cv_results_ for the best param combo.

    cv_results_ has keys like: 'mean_test_accuracy', 'std_test_accuracy', 'mean_test_f1', etc.
    Returns shape: {"accuracy": {"mean": ..., "std": ...}, ...} to match log_model_to_wandb's expectation.
    """
    cv_scores = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        cv_scores[metric] = {
            "mean": cv_results[f"mean_test_{metric}"][best_index],
            "std": cv_results[f"std_test_{metric}"][best_index],
        }
    return cv_scores


def main():
    print("=" * 70)
    print("GridSearchCV Hyperparameter Tuning: SVM (char n-grams, C parameter)")
    print("=" * 70)
    print()

    # Load configuration and data
    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    df, df_raw, text_column, label_column = load_training_data(config, project_root)
    project_name = config["project"]["name"]

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

    # Build pipeline: TF-IDF (char analyzer) + SVM
    print("Building pipeline: TF-IDF(char) → SVM...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char")),
            (
                "svm",
                SVC(
                    kernel="linear", class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )

    # Define hyperparameter grid
    param_grid = {
        "tfidf__ngram_range": [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5)],
        "svm__C": [0.01, 0.1, 1, 10, 100],
    }
    print(f"Parameter grid: {param_grid}\n")

    # Create GridSearchCV with multiple scorers (to track all metrics per fold)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        refit="f1",  # Optimize for F1, the project's key metric
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )

    print("Performing GridSearchCV (5-fold stratified CV)...")
    print("(This may take a few minutes...)\n")
    grid_search.fit(X_train, y_train)
    print()

    # Report best hyperparameters and CV score
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print("Best parameters:")
    print(f"  - TF-IDF n-gram range: {best_params['tfidf__ngram_range']}")
    print(f"  - SVM C: {best_params['svm__C']}")
    print(f"Best CV F1 score: {best_cv_f1:.4f}\n")

    # Extract CV metrics for the best param combination
    best_index = grid_search.best_index_
    cv_scores = extract_cv_scores_for_best_params(grid_search.cv_results_, best_index)
    print("Best params CV metrics:")
    for metric, values in cv_scores.items():
        print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print()

    # Evaluate on test set
    print("Evaluating best model on test set...")
    metrics, cm, report = evaluate(grid_search.best_estimator_, X_test, y_test)
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

    # Build descriptive model name
    ngram_range_str = (
        f"{best_params['tfidf__ngram_range'][0]}-{best_params['tfidf__ngram_range'][1]}"
    )
    c_str = str(best_params["svm__C"]).replace(".", "_")
    model_name = f"SVM_gridsearch_char_ngram_{ngram_range_str}_C_{c_str}"
    predictions_path = get_predictions_path(config, project_root, model_name)

    # Save predictions
    print("Saving model predictions...")
    y_pred = grid_search.best_estimator_.predict(X_test)
    save_predictions(X_test, X_test_raw, y_test, y_pred, predictions_path)
    print(f"Predictions saved to {predictions_path}\n")

    # Log to Weights & Biases
    print("Logging to Weights & Biases...")
    preprocess_steps = config["task"]["preprocessing_steps"]

    wandb.init(
        project=project_name,
        name=model_name,
        config={
            "model": model_name,
            "preprocessing": preprocess_steps,
            "pipeline": "TF-IDF(char) + SVM",
            "best_tfidf_ngram_range": best_params["tfidf__ngram_range"],
            "best_svm_C": best_params["svm__C"],
            "svm_kernel": "linear",
            "svm_class_weight": "balanced",
            "gridsearch_param_grid": {
                "tfidf__ngram_range": [(1, 3), (1, 4), (1, 5), (2, 4)],
                "svm__C": [0.01, 0.1, 1, 10, 100],
            },
            "cv_folds": 5,
            "cv_random_state": 42,
        },
    )

    # Log CV metrics for best params
    wandb.log({"cv_accuracy_mean": cv_scores["accuracy"]["mean"]})
    wandb.log({"cv_accuracy_std": cv_scores["accuracy"]["std"]})
    wandb.log({"cv_precision_mean": cv_scores["precision"]["mean"]})
    wandb.log({"cv_precision_std": cv_scores["precision"]["std"]})
    wandb.log({"cv_recall_mean": cv_scores["recall"]["mean"]})
    wandb.log({"cv_recall_std": cv_scores["recall"]["std"]})
    wandb.log({"cv_f1_mean": cv_scores["f1"]["mean"]})
    wandb.log({"cv_f1_std": cv_scores["f1"]["std"]})

    # Log test metrics
    wandb.log({"test_accuracy": metrics["accuracy"]})
    wandb.log({"test_precision": metrics["precision"]})
    wandb.log({"test_recall": metrics["recall"]})
    wandb.log({"test_f1": metrics["f1"]})

    wandb.finish()
    print("Logged to W&B!\n")

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
