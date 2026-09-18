from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import evaluate_model
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

def build_model() -> Pipeline:
    """Build the baseline TF-IDF + Logistic Regression model."""

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    model = build_model()

    print("\nTraining baseline model...")
    model.fit(X_train, y_train)

    print("Training complete.")

    predictions = model.predict(X_test)

    print("\nMODEL PERFORMANCE")
    evaluate_model(y_test, predictions)
