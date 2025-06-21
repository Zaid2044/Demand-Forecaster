import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print("--- Starting ADVANCED Model Training ---")

print("Loading data...")
train_df = pd.read_csv('data/train.csv', parse_dates=['Date'], low_memory=False)
store_df = pd.read_csv('data/store.csv')
df = pd.merge(train_df, store_df, on='Store', how='left')
print("Data loaded and merged successfully.")

print("Cleaning and preprocessing data...")
df = df[df['Sales'] > 0]
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
df.fillna(0, inplace=True)

print("Performing advanced feature engineering...")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

df.sort_values(['Store', 'Date'], inplace=True)

for lag in [1, 7, 14]:
    df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

df['sales_rolling_mean_7'] = df.groupby('Store')['Sales'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['sales_rolling_mean_30'] = df.groupby('Store')['Sales'].rolling(window=30).mean().reset_index(level=0, drop=True)

df.fillna(0, inplace=True)

print("Preparing data for modeling...")
df['StateHoliday'] = pd.Categorical(df['StateHoliday'].astype(str), categories=['0', 'a', 'b', 'c'], ordered=False)
df['StoreType'] = pd.Categorical(df['StoreType'], categories=['a', 'b', 'c', 'd'], ordered=False)
df['Assortment'] = pd.Categorical(df['Assortment'], categories=['a', 'b', 'c'], ordered=False)

categorical_features = ['StoreType', 'Assortment', 'StateHoliday']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

features = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'CompetitionDistance',
    'Year', 'Month', 'Day', 'WeekOfYear',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14',
    'sales_rolling_mean_7', 'sales_rolling_mean_30',
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

print("Training LightGBM model...")
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

print("Evaluating model performance...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- ADVANCED Model Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("--------------------------------")

print("Saving the trained model...")
model_path = 'models/advanced_forecaster_model.joblib'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")
print("\n--- Script Finished ---")