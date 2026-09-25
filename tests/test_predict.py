import pytest

from tellascam.predict import load_model, predict_message


@pytest.fixture
def model():
    return load_model()


def test_prediction_returns_valid_label(model):
    prediction, probability = predict_message(
        model,
        "Are we still meeting tomorrow?"
    )

    assert prediction in ["ham", "spam"]


def test_probability_is_valid(model):
    _, probability = predict_message(
        model,
        "Congratulations! Claim your free prize now."
    )

    assert 0.0 <= probability <= 1.0