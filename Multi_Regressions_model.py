import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
import xgboost as xgb

# -----------------------------
# Setup: ensure folders exist
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv('data/avocado.csv')

cols_to_drop = ['Unnamed: 0', '4046', '4225', '4770', 'Date']

df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

df.drop(['Unnamed: 0', '4046', '4225', '4770', 'Date'], axis=1, inplace=True)


# Encode categorical columns
le_type = LabelEncoder()
le_region = LabelEncoder()
df['type'] = le_type.fit_transform(df['type'])
df['region'] = le_region.fit_transform(df['region'])

# Save label encoders
joblib.dump(le_type, 'models/label_encoder_type.pkl')
joblib.dump(le_region, 'models/label_encoder_region.pkl')

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop('AveragePrice', axis=1)
y = df['AveragePrice']
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -----------------------------
# 🤖 Define regression models
# -----------------------------
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

# -----------------------------
# Train, evaluate, and save
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    })

    joblib.dump(model, f'models/{name}.pkl')

# -----------------------------
# Save evaluation report
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv('models/results.csv', index=False)

print("All models trained and saved. Evaluation metrics stored in models/results.csv.")

