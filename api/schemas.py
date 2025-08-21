from pydantic import BaseModel

class PredictionResponse(BaseModel):
    predicted_class: str
    probability_pneumonia: float
    probability_normal: float
    threshold: float
    metadata: dict
