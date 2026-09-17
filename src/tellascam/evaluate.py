from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred):
    """Evaluate classification performance."""

    accuracy = accuracy_score(y_true, y_pred)

    print("\nMODEL EVALUATION")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred))

    print("CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))