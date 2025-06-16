# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import io
import base64

sns.set(style="whitegrid")

def plot_feature_distribution(df):
    """Plot distribution of features colored by flare label."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    features = ["Sunspot Number", "Radio Flux", "X-Ray Emission"]
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue="Flare", multiple="stack", ax=axs[i], palette="Set1")
        axs[i].set_title(f"Distribution of {feature} by Flare Label")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix for {model_name}")
    return fig

def plot_roc_curve(model, X_test, y_test, model_name):
    """Plot ROC curve for a model."""
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    plt.close(fig)
    return encoded
