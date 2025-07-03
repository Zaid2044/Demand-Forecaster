<h1 align="center">🚀 Demand Forecaster API</h1>
<p align="center">
  An intelligent, production-ready API for predicting store sales using LightGBM and FastAPI.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-8BC34A?style=flat&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Uvicorn-333333?style=flat"/>
</p>

---

## 🧠 Overview

**Demand Forecaster** is a full-stack ML project that predicts store sales using advanced time-series features and a LightGBM model. Built with a clean architecture, it offers a blazing-fast REST API using FastAPI and interactive documentation via Swagger UI.

This project demonstrates a complete machine learning workflow:
- 📊 Data exploration
- 🔧 Feature engineering
- 🧠 Model training
- 🌐 API deployment

---

## ✨ Features

- ⚡ **R² score > 0.90** with optimized LightGBM model
- 🕰️ Time-series lag features and rolling averages
- ⚙️ Live REST API with FastAPI + Uvicorn
- 📄 Auto-generated Swagger docs (`/docs`)
- 📂 Clean project structure with `data/`, `models/`, `src/`, and `notebooks/`

---

## 🔍 Tech Stack

- **Languages:** Python 3.9+
- **Libraries:** pandas, scikit-learn, LightGBM, matplotlib, seaborn, joblib
- **API:** FastAPI, Uvicorn
- **Tools:** Jupyter Notebook

---

## 🚀 Performance

| Metric | Value |
|--------|-------|
| R²     | **0.91** |
| MSE    | ~838,193 |

The model explains 91% of the variance in the dataset — solid for retail forecasting.

---

## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.9+
- Git
- Kaggle account to download dataset

### 📦 Installation

```bash
git clone https://github.com/Zaid2044/Demand-Forecaster.git
cd Demand-Forecaster
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Create requirements.txt with:

```nginx
Copy
Edit
pandas
scikit-learn
matplotlib
seaborn
jupyter
joblib
lightgbm
"fastapi[all]"
```

Then:

```bash
pip install -r requirements.txt
```

### 📥 Dataset
Download from Kaggle: Rossmann Store Sales

Place train.csv and store.csv inside the data/ folder

## 🔧 How to Use
### 1. 📊 Data Exploration
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. 🧠 Train the Model
```bash
python src/train.py
```
Creates: models/advanced_forecaster_model.joblib

### 3. 🌐 Launch the API
```bash
uvicorn src.api:app --reload
```

Visit http://127.0.0.1:8000

Explore Swagger docs at http://127.0.0.1:8000/docs

## 🧑‍💻 Author
**Zaid Ahmed**
