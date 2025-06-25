# 🧠 Multi-Regression Framework for Price Prediction

A robust machine learning framework that implements and evaluates **13 regression models** to predict market prices with high accuracy, integrated into a dynamic **Flask web application** for real-time inference.

---

## 🚀 Overview

This project is a full-cycle machine learning system for price prediction based on structured data (originally applied to avocado pricing). It covers:

- Data preprocessing with label encoding
- Training and evaluation of diverse regression models
- Model performance visualization
- Web-based interface for real-time prediction and model comparison

---

## 📊 Models Implemented

| Model Type            | ML Technique Used               |
|-----------------------|---------------------------------|
| Linear                | Linear Regression               |
| Robust                | Huber Regressor                 |
| Regularized           | Ridge, Lasso, ElasticNet        |
| Polynomial            | Polynomial Regression (deg=2)  |
| Optimization-based    | SGD Regressor                   |
| Neural Network        | MLPRegressor (ANN)              |
| Ensemble              | Random Forest, LightGBM, XGBoost|
| Distance-based        | K-Nearest Neighbors             |
| Kernel-based          | Support Vector Regression       |

Each model is trained, evaluated using MAE, MSE, and R² Score, and saved as a `.pkl` file for production-ready inference.

---

## 🧰 Tech Stack

- **Python** (Pandas, scikit-learn, XGBoost, LightGBM)
- **Flask** (for interactive web UI)
- **HTML + Bootstrap** (UI rendering)
- **Joblib** (model persistence)

---

## 💡 Features

✅ Trains and benchmarks 13 regression models  
✅ Supports dynamic model selection for inference  
✅ Provides intuitive web UI to input test data  
✅ Outputs real-time price prediction  
✅ Displays model performance comparison  

---

## 🖥️ Demo

### 🔎 Predict Price
- Select any trained model
- Input volume, bag type, region, and year
- Receive instant prediction on-screen

### 📈 View Evaluation
- Compare all models on MAE, MSE, R²
- Identify top-performing regressors

---

## 🧪 Sample Input Format

| Feature         | Example        |
|----------------|----------------|
| Total Volume   | 500000.0       |
| Total Bags     | 300000.0       |
| Small Bags     | 100000.0       |
| Large Bags     | 150000.0       |
| XLarge Bags    | 50000.0        |
| Type           | conventional   |
| Region         | Albany         |
| Year           | 2017           |

---

## 📦 Setup Instructions

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Train models:
  ```
  python Multi_Regressions_model.py
  ```
4. Run the Flask app:
  ```
  python Multi-regressions.py
  ```

---
## Thank You
---
  
