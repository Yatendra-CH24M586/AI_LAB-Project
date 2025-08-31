# api/app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd

app = FastAPI()

# Load model from registry
MODEL_NAME = "TitanicRF"
STAGE = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")


class Passenger(BaseModel):
    Pclass: int
    Sex_idx: float
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_idx: float
    FamilySize: int
    IsAlone: int
    Title_idx: float


@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.dict()])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
