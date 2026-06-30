from sklearn.svm import SVC
from sklearn.base import clone
from src.utils import load_config, get_config_path, get_project_root, load_csv
from src.train_model import (
    split_dataset,
    vectorize_text,
    cross_validate
)
from src.evaluate import evaluate


def main():
    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    local_preprocessed_data_path = (project_root / config["data"]["local_preprocessed_data_path"]).resolve()
    text_column = config["task"]["text_column"]
    label_column = config["task"]["label_column"]

    test_size = config["reproducibility"]["train_test_split"]["test_size"]
    random_state = config["reproducibility"]["train_test_split"]["random_state"]
    stratify = config["reproducibility"]["train_test_split"]["stratify"]

    print(f"Loading preprocessed data from {local_preprocessed_data_path}...")
    df = load_csv(str(local_preprocessed_data_path))
    print(df.head())
    print(f"Loaded {len(df)} rows\n")

    X = df[text_column].values
    y = df[label_column].values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}\n")

    print("Vectorizing text...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(
        X_train, X_test,
        ngram_range=(1, 1)
    )
    print(f"TF-IDF shape: {X_train_tfidf.shape}\n")

    print("Creating SVM model...")
    model = SVC(kernel="linear", random_state=random_state)
    print(f"Model: {model}\n")

    print("Performing cross-validation...")
    cv_scores = cross_validate(clone(model), X_train_tfidf, y_train, cv_folds=5)
    print("Cross-validation scores:")
    for metric, score in cv_scores.items():
        print(f"  {metric}: {score:.4f}")
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

    print("Done!")


if __name__ == "__main__":
    main()
