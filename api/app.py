
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import logging

app = FastAPI()

# Load model dan preprocessing
model = joblib.load("./models/model.pkl")
scaler = joblib.load("./models/preprocessing.pkl")

# Definisikan schema input menggunakan Pydantic
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Setup logging
logging.basicConfig(
    filename="./logs/inference.log",  # Simpan di folder logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    amount_scaled = scaler.transform([[data["Amount"]]])[0][0]
    features = [data[f"V{i}"] for i in range(1, 29)] + [amount_scaled]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    # Log inference
    logging.info(f"Request: {data} | Prediction: {prediction} | Probability: {round(probability, 4)}")

    return {"prediction": int(prediction), "probability": round(float(probability), 4)}
