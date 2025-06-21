# 🚀 Advanced Sales Forecaster API

An end-to-end data science project that predicts store sales using a high-performance LightGBM model. The project includes advanced feature engineering with time-series data and is deployed as a live, interactive web API using FastAPI.

This project demonstrates a full machine learning lifecycle: data exploration, advanced feature engineering, model training, and production-ready API deployment.

---

## ✨ Key Features

-   **High-Performance Model:** Utilizes **LightGBM**, a state-of-the-art gradient boosting framework, achieving an **R² score of over 0.90**.
-   **Advanced Feature Engineering:** Incorporates time-series features like sales lags and rolling averages, dramatically improving predictive accuracy.
-   **Live API Deployment:** The trained model is served via a **FastAPI** backend, making it a usable and scalable tool.
-   **Interactive Documentation:** The API includes automatically generated, interactive documentation (via Swagger UI) for easy testing and integration.
-   **Professional Structure:** The project is organized with a clear and professional data science structure, separating data, notebooks, source code, and saved models.

---

## 🛠️ Technology Stack

-   **Python**
-   **Data Science:** Pandas, Scikit-learn, Matplotlib, Seaborn
-   **Machine Learning:** LightGBM
-   **API Framework:** FastAPI
-   **Server:** Uvicorn
-   **Tooling:** Jupyter, Joblib

---

## 📈 Performance

The advanced model with time-series features achieves a significantly higher performance compared to a baseline model.

-   **R-squared (R²): 0.91**
-   **Mean Squared Error (MSE):** ~838,193

This indicates the model can explain approximately 91% of the variance in the sales data.

---

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Zaid2044/Demand-Forecaster.git
    cd Demand-Forecaster
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is recommended for professional projects. Create it and add the following:
    ```
    pandas
    scikit-learn
    matplotlib
    seaborn
    jupyter
    joblib
    lightgbm
    "fastapi[all]"
    ```
    Then, install from the file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    -   Download the dataset from the [Rossmann Store Sales Kaggle page](https://www.kaggle.com/c/rossmann-store-sales/data).
    -   Unzip the file.
    -   Place `train.csv` and `store.csv` inside the `data/` folder.

---

## ⚡ Usage

The project is split into three main parts: exploration, training, and API deployment.

### 1. (Optional) Explore the Data

The `notebooks/EDA.ipynb` file contains the initial data exploration and visualization. To run it:
```bash
jupyter notebook

2. Train the Advanced Model

Run the training script to process the data, perform feature engineering, and train the LightGBM model. This will create a advanced_forecaster_model.joblib file in the models/ directory.

python src/train.py

3. Run the API Server

Deploy the trained model as a live API.
uvicorn src.api:app --reload

The server will be running on http://127.0.0.1:8000.
Go to http://127.0.0.1:8000/docs in your browser to access the interactive API documentation and make test predictions.