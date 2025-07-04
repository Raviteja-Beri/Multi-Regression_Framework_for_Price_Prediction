import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

# Set MLflow Tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Multi-Regression Price Prediction")

# Load evaluation results
results_df = pd.read_csv("models/results.csv")

# Loop through each model entry
for _, row in results_df.iterrows():
    model_name = row['Model']
    mae = row['MAE']
    mse = row['MSE']
    r2 = row['R2']
    
    model_path = f"models/{model_name}.pkl"

    if not os.path.exists(model_path):
        print(f"[!] Skipping {model_name}: model file not found.")
        continue

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Load the model
        model = joblib.load(model_path)

        # Log metrics
        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "R2_Score": r2
        })

        # Log model artifact
        mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)

        print(f"[✓] Logged {model_name} to MLflow.")
