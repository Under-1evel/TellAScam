from fastapi import FastAPI
from pydantic import BaseModel

from .predict import load_model, predict_message


app = FastAPI(
    title="TellAScam API",
    description="Machine-learning API for detecting suspicious SMS messages.",
    version="1.0.0",
)

model = load_model()


class MessageRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    prediction: str
    spam_probability: float


@app.get("/")
def root():
    return {
        "name": "TellAScam API",
        "status": "running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    prediction, probability = predict_message(
        model,
        request.message
    )

    return PredictionResponse(
        prediction=prediction,
        spam_probability=probability
    )