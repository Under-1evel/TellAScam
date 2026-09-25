from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from .preprocessing import load_data
from .evaluate import evaluate_model, analyze_errors

MODEL_PATH = Path("models/tellascam_model.joblib")

def save_model(model):
    """Save a trained TellAScam model to disk."""

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

def split_data():
    data = load_data()

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

def build_model(
    balanced=False,
    ngram_range=(1, 1)
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression model."""

    class_weight = "balanced" if balanced else None

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight=class_weight
            )
        )
    ])

    return model

def cross_validate_model(model, X_train, y_train):
    """Evaluate a model using 5-fold stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    y_binary = (y_train == "spam").astype(int)

    results = cross_validate(
        model,
        X_train,
        y_binary,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print(f"Precision: {np.mean(results['test_precision']):.4f}")
    print(f"Recall:    {np.mean(results['test_recall']):.4f}")
    print(f"F1 Score:  {np.mean(results['test_f1']):.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Experiment 1: Original baseline
    baseline_model = build_model(balanced=False)

    print("\nTraining baseline model...")
    baseline_model.fit(X_train, y_train)

    baseline_predictions = baseline_model.predict(X_test)

    print("\nBASELINE MODEL")
    evaluate_model(y_test, baseline_predictions)

    # Experiment 2: Balanced class weights
    balanced_model = build_model(balanced=True)

    print("\nTraining balanced model...")
    balanced_model.fit(X_train, y_train)

    balanced_predictions = balanced_model.predict(X_test)

    print("\nBALANCED MODEL")
    evaluate_model(y_test, balanced_predictions)

    print("\nBALANCED MODEL ERROR ANALYSIS")
    analyze_errors(
    X_test,
    y_test,
    balanced_predictions
    )

    print("\nCROSS-VALIDATION EXPERIMENT")

    print("\nMODEL A: UNIGRAMS")
    unigram_model = build_model(
        balanced=True,
        ngram_range=(1, 1)
    )

    cross_validate_model(
        unigram_model,
        X_train,
        y_train
    )

    print("\nMODEL B: UNIGRAMS + BIGRAMS")
    bigram_model = build_model(
        balanced=True,
        ngram_range=(1, 2)
    )

    cross_validate_model(
        bigram_model,
        X_train,
        y_train
    )

    print("\nTRAINING SELECTED MVP MODEL")

    final_model = build_model(
        balanced=True,
        ngram_range=(1, 2)
    )

    final_model.fit(X_train, y_train)

    save_model(final_model)
