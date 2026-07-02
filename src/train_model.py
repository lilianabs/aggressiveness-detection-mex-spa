import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.evaluate import evaluate


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
