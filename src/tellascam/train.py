from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import evaluate_model, analyze_errors
from preprocessing import load_data

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

def build_model(balanced=False) -> Pipeline:
    """Build a TF-IDF + Logistic Regression model."""

    class_weight = "balanced" if balanced else None

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
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
