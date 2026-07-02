import os
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from src.utils import load_config, get_config_path, get_project_root, load_csv
from src.train_model import split_dataset, vectorize_text, cross_validate
from src.evaluate import evaluate
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)


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
    local_preprocessed_data_path = (
        project_root / config["data"]["local_preprocessed_data_path"]
    ).resolve()
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
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}\n")

    print("Vectorizing text...")
    X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test, ngram_range=(1, 3))
    print(f"TF-IDF shape: {X_train_tfidf.shape}\n")

    preprocess_steps = config["task"]["preprocessing_steps"]

    for model_name, model in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"Training {model_name}...")
        print(f"{'=' * 50}")

        print("\nPerforming cross-validation...")
        cv_scores = cross_validate(clone(model), X_train_tfidf, y_train, cv_folds=5)
        print("Cross-validation scores:")
        for metric, values in cv_scores.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        print()

        print(f"Training {model_name} on full train set...")
        model.fit(X_train_tfidf, y_train)
        print(f"{model_name} trained!\n")

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

        print(f"Logging {model_name} to Weights & Biases...")
        wandb.init(
            project="aggressiveness-detection",
            name=model_name,
            config={
                "model": model_name + " with class_weight=balanced",
                "Preprocessing": preprocess_steps,
                "TF-IDF": "TF-IDF vectorization with n-grams (1, 3)",
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

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
