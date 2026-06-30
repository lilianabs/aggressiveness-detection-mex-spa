import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return metrics, cm, report


def save_predictions(X_test_text, X_test_text_raw, y_test, y_pred, output_path):
    df = pd.DataFrame({
        "Text": X_test_text,
        "Text_raw": X_test_text_raw,
        "Category": y_test,
        "Prediction": y_pred,
    })
    df.to_csv(output_path, index=False)
