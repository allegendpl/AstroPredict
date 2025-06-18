# src/evaluation.py

import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "auc": auc,
        "report": report,
    }

def evaluate_all(models, X_test, y_test):
    """Evaluate all models and return a summary."""
    results = {}
    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        print(f" {name}")
        print(f"Accuracy: {res['accuracy']}")
        print(f"AUC: {res['auc']}")
        print(f"Classification Report:\n{res['report']}")

    return results

def save_report(results, directory='models/trained_models'):
    """Save evaluation results to disk."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "evaluation_report.txt")
    with open(path, "w") as f:
        for name, res in results.items():
            f.write(f"{name}\n")
            f.write(f"Accuracy: {res['accuracy']}\n")
            f.write(f"AUC: {res['auc']}\n")
            f.write(f"Report:\n{res['report']}\n\n")
    print(f" Evaluation report saved to {path}")
