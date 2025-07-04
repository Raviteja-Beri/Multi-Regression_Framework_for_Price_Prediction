from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 
    'XGBoost', 'KNN'
]
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = {
    name: joblib.load(os.path.join(BASE_DIR, 'models', f'{name}.pkl'))
    for name in model_names
}

# Load feature names and evaluation results
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
results_df = pd.read_csv(os.path.join(BASE_DIR, 'models', 'results.csv'))

# Define categorical mappings (must match the label encoder output used in training)
type_options = {'conventional': 0, 'organic': 1}
region_options = {
    'Albany': 0, 'Atlanta': 1, 'BaltimoreWashington': 2, 'Boise': 3, 'Boston': 4,
    'BuffaloRochester': 5, 'California': 6, 'Charlotte': 7, 'Chicago': 8, 'CincinnatiDayton': 9,
    'Columbus': 10, 'DallasFtWorth': 11, 'Denver': 12, 'Detroit': 13, 'GrandRapids': 14,
    'GreatLakes': 15, 'HarrisburgScranton': 16, 'HartfordSpringfield': 17, 'Houston': 18,
    'Indianapolis': 19, 'Jacksonville': 20, 'LasVegas': 21, 'LosAngeles': 22, 'Louisville': 23,
    'MiamiFtLauderdale': 24, 'Midsouth': 25, 'Nashville': 26, 'NewOrleansMobile': 27,
    'NewYork': 28, 'Northeast': 29, 'NorthernNewEngland': 30, 'Orlando': 31, 'Philadelphia': 32,
    'PhoenixTucson': 33, 'Pittsburgh': 34, 'Plains': 35, 'Portland': 36, 'RaleighGreensboro': 37,
    'RichmondNorfolk': 38, 'Roanoke': 39, 'Sacramento': 40, 'SanDiego': 41, 'SanFrancisco': 42,
    'Seattle': 43, 'SouthCarolina': 44, 'SouthCentral': 45, 'Southeast': 46, 'Spokane': 47,
    'StLouis': 48, 'Syracuse': 49, 'Tampa': 50, 'TotalUS': 51, 'West': 52, 'WestTexNewMexico': 53
}

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names, type_options=type_options, region_options=region_options)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    
    input_data = {
        'Total Volume': float(request.form['Total Volume']),
        'Total Bags': float(request.form['Total Bags']),
        'Small Bags': float(request.form['Small Bags']),
        'Large Bags': float(request.form['Large Bags']),
        'XLarge Bags': float(request.form['XLarge Bags']),
        'type': type_options[request.form['type']],
        'region': region_options[request.form['region']],
        'year': float(request.form['year'])
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
