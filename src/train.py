# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("--- Starting Model Training ---")

print("Loading data...")
try:
    train_df = pd.read_csv('data/train.csv', low_memory=False)
    store_df = pd.read_csv('data/store.csv')
    df = pd.merge(train_df, store_df, on='Store', how='left')
    print("Data loaded and merged successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you are running this script from the project root directory.")
    exit()

print("Cleaning and preprocessing data...")
df = df[(df['Open'] == 1) & (df['Sales'] > 0)]
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
df.fillna(0, inplace=True)

print("Performing feature engineering...")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

def is_promo_active(row):
    promo_months = str(row['PromoInterval']).split(',')
    if row['Date'].strftime('%b') in promo_months:
        return 1
    return 0

if 'PromoInterval' in df.columns:
    df['IsPromoMonth'] = df.apply(is_promo_active, axis=1)

print("Preparing data for modeling...")
df['StateHoliday'] = pd.Categorical(df['StateHoliday'].astype(str), categories=['0', 'a', 'b', 'c'], ordered=False)
df['StoreType'] = pd.Categorical(df['StoreType'], categories=['a', 'b', 'c', 'd'], ordered=False)
df['Assortment'] = pd.Categorical(df['Assortment'], categories=['a', 'b', 'c'], ordered=False)

categorical_features = ['StoreType', 'Assortment', 'StateHoliday']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

features = [
    'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsPromoMonth',
    'StoreType_b', 'StoreType_c', 'StoreType_d',
    'Assortment_b', 'Assortment_c',
    'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'
]
target = 'Sales'

existing_features = [f for f in features if f in df.columns]
X = df[existing_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

print("Training RandomForest Regressor model...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
model.fit(X_train, y_train)
print("Model training complete.")

print("Evaluating model performance...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("--------------------------------")

print("Saving the trained model...")
model_path = 'models/demand_forecaster_model.joblib'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")
print("\n--- Script Finished ---")