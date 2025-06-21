# src/predict.py

import pandas as pd
import joblib

print("--- Starting Prediction Script ---")

print("Loading trained model and store data...")
try:
    model = joblib.load('models/demand_forecaster_model.joblib')
    store_df = pd.read_csv('data/store.csv')
    print("Model and store data loaded.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the model and data files exist and you are running from the project root.")
    exit()

def predict_sales(new_data):
    """
    Preprocesses new data and predicts sales using the trained model.
    """
    print("\nProcessing new data for prediction...")
    
    df = pd.merge(new_data, store_df, on='Store', how='left')

    df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace=True)
    df.fillna(0, inplace=True)

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

    df['StateHoliday'] = pd.Categorical(df['StateHoliday'].astype(str), categories=['0', 'a', 'b', 'c'], ordered=False)
    df['StoreType'] = pd.Categorical(df['StoreType'], categories=['a', 'b', 'c', 'd'], ordered=False)
    df['Assortment'] = pd.Categorical(df['Assortment'], categories=['a', 'b', 'c'], ordered=False)
    
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    model_features = [
        'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
        'CompetitionDistance', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsPromoMonth',
        'StoreType_b', 'StoreType_c', 'StoreType_d',
        'Assortment_b', 'Assortment_c',
        'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'
    ]
    
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
            
    df_aligned = df[model_features]

    print("Making predictions...")
    predictions = model.predict(df_aligned)
    return predictions

if __name__ == "__main__":
    example_data = pd.DataFrame({
        'Store': [1, 8],
        'DayOfWeek': [4, 4],  
        'Date': ['2015-08-01', '2015-08-01'],
        'Open': [1, 1],
        'Promo': [1, 1],
        'StateHoliday': ['0', '0'], 
        'SchoolHoliday': [1, 1]
    })
    
    predicted_sales = predict_sales(example_data)
    
    print("\n--- Prediction Results ---")
    for i, prediction in enumerate(predicted_sales):
        store_id = example_data['Store'].iloc[i]
        date = example_data['Date'].iloc[i]
        print(f"Predicted sales for Store {store_id} on {date}: ₹{prediction:.2f}")
    
    print("\n--- Script Finished ---")