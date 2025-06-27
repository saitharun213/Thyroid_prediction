from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Union

app = FastAPI(title="Thyroid Disease Prediction API")

# Load the model
model = joblib.load('thyroid_model.pkl')

# Define input schema using Pydantic
class ThyroidInput(BaseModel):
    age: int
    sex: str
    on_thyroxine: str
    query_on_thyroxine: str
    on_antithyroid_meds: str
    sick: str
    pregnant: str
    thyroid_surgery: str
    I131_treatment: str
    query_hypothyroid: str
    query_hyperthyroid: str
    lithium: str
    goitre: str
    tumor: str
    hypopituitary: str
    psych: str
    TSH_measured: str
    TSH: Union[float, None]
    T3_measured: str
    T3: Union[float, None]
    TT4_measured: str
    TT4: Union[float, None]
    T4U_measured: str
    T4U: Union[float, None]
    FTI_measured: str
    FTI: Union[float, None]
    TBG_measured: str
    referral_source: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Thyroid Disease Prediction API"}

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: ThyroidInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)