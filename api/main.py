from fastapi import FastAPI
from .Features import Features
import pandas as pd
import joblib
from pathlib import Path

path = Path(__file__).resolve().parents[1]

model = joblib.load(f"{path}/models/model.joblib")

class Model():

    app = FastAPI()

    @app.post("/predict")
    def predict(data: Features):
            df = pd.DataFrame([data.model_dump()])
            prediction = model.predict(df)
            prediction_value = float(prediction[0])
            return {
                "prediction":prediction_value
            }