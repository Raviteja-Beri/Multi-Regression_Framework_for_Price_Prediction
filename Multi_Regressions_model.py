import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
dataset = pd.read_csv('data/avocado.csv')
dataset.drop(['Unnamed: 0','4046','4225','4770','Date'], axis=1, inplace=True)

# Label encoding
le_type = LabelEncoder()
le_region = LabelEncoder()
dataset['type'] = le_type.fit_transform(dataset['type'])
dataset['region'] = le_region.fit_transform(dataset['region'])

# Save the label encoders if needed later
joblib.dump(le_type, 'label_encoder_type.pkl')
joblib.dump(le_region, 'label_encoder_region.pkl')

# Prepare features and target
X = dataset.drop('AveragePrice', axis=1)
y = dataset['AveragePrice']

# Save feature names AFTER encoding
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

results = []

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    })

    joblib.dump(model, f'{name}.pkl')

# Save evaluation metrics
results_df = pd.DataFrame(results)
results_df.to_csv('Multi-Regression Framework for Price Prediction.csv', index=False)

print("Models trained and saved. Evaluation saved to CSV.")
