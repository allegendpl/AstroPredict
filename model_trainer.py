# src/model_trainer.py

import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data

def split_data(data, test_ratio=0.2):
    """Split data into training and testing sets."""
    X = data[["Sunspot Number", "Radio Flux", "X-Ray Emission"]]
    y = data["Flare"]
    return train_test_split(X, y, test_size=test_ratio, random_state=42)

def train_models(X_train, y_train):
    """Train Logistic Regression, SVC, and RandomForest and return trained models."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "SVC": SVC(probability=True),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f" {name} trained.")
    return trained

def cross_validate_models(models, X, y):
    """Perform 5-fold cross-validation and return scores."""
    scores = {}
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        scores[name] = score.mean()
        print(f"{name} CV Accuracy: {score.mean():.4f}")
    return scores

def save_models(models, directory='models/trained_models'):
    """Save trained models to disk."""
    os.makedirs(directory, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(directory, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f" {name} saved to {path}")
