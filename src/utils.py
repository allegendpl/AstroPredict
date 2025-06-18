import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot the top N feature importances from a trained model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    return plt

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Generate a labeled confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return plt

def display_classification_report(y_true, y_pred):
    """
    Return classification report as a formatted DataFrame.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report
