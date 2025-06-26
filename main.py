from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

model = load_model("account_classification_model.h5")
scaler = joblib.load("scaler.pkl")

class AccountFeatures(BaseModel):
    edge_followed_by: int
    edge_follow: int
    username_length: int
    username_has_number: int
    full_name_length: int
    is_private: int
    is_joined_recently: int
    has_channel: int
    has_guides: int
    has_external_url: int
    has_highlight_reels: int
    is_business_account: int

@app.post("/predict")
def predict(account: AccountFeatures):
    data = np.array([[ 
        account.edge_followed_by,
        account.edge_follow,
        account.username_length,
        account.username_has_number,
        account.full_name_length,
        account.is_private,
        account.is_joined_recently,
        account.has_channel,
        account.has_guides,
        account.has_external_url,
        account.has_highlight_reels,
        account.is_business_account
    ]])
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)
    label = "Fake (Reject)" if prediction[0][0] > 0.5 else "Real (Accept)"
    return {"prediction_score": float(prediction[0][0]), "result": label}
