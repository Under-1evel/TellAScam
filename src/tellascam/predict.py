from pathlib import Path

import joblib


MODEL_PATH = Path("models/tellascam_model.joblib")


def load_model():
    """Load the trained TellAScam model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_message(model, message):
    """Predict whether a message is spam."""

    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    classes = model.classes_
    spam_index = list(classes).index("spam")
    spam_probability = probabilities[spam_index]

    return prediction, spam_probability


if __name__ == "__main__":
    model = load_model()

    message = input("Enter a message to analyze: ")

    prediction, spam_probability = predict_message(
        model,
        message
    )

    print(f"\nPrediction: {prediction.upper()}")
    print(f"Spam probability: {spam_probability:.2%}")