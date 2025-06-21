from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Sales Forecaster API",
    description="An API to predict store sales using a LightGBM model.",
    version="1.0"
)

class StoreFeatures(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    CompetitionDistance: float
    Year: int
    Month: int
    Day: int
    WeekOfYear: int
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_rolling_mean_7: float
    sales_rolling_mean_30: float
    StoreType: str
    Assortment: str
    StateHoliday: str

@app.on_event("startup")
def load_model():
    global model
    model_path = "models/advanced_forecaster_model.joblib"
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        model = None

@app.post("/predict/")
def predict_sales(features: StoreFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check the server logs."}

    input_df = pd.DataFrame([features.model_dump()])

    input_df['StateHoliday'] = pd.Categorical(input_df['StateHoliday'], categories=['0', 'a', 'b', 'c'])
    input_df['StoreType'] = pd.Categorical(input_df['StoreType'], categories=['a', 'b', 'c', 'd'])
    input_df['Assortment'] = pd.Categorical(input_df['Assortment'], categories=['a', 'b', 'c'])

    input_df = pd.get_dummies(input_df, drop_first=True)

    model_features = [
        'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'CompetitionDistance',
        'Year', 'Month', 'Day', 'WeekOfYear',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14',
        'sales_rolling_mean_7', 'sales_rolling_mean_30',
        'StoreType_b', 'StoreType_c', 'StoreType_d',
        'Assortment_b', 'Assortment_c',
        'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'
    ]

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df_aligned = input_df[model_features]

    prediction = model.predict(input_df_aligned)
    
    return {"predicted_sales": round(prediction[0], 2)}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecaster API. Go to /docs for the interactive API documentation."}