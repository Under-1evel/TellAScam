from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred):
    """Evaluate classification performance."""

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="spam"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="spam"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="spam"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["ham", "spam"]
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)