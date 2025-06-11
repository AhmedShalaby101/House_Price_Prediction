California Housing Price Prediction
https://img.shields.io/badge/License-MIT-yellow.svg https://img.shields.io/badge/Python-3.8+-blue.svg

Project Description
Machine learning pipeline to predict California housing prices using demographic and geographic features. Processes raw data, engineers features, trains multiple models, and evaluates performance.

Features
Data cleaning (handles 207 missing values in total_bedrooms)

One-Hot Encoding for ocean_proximity (5 categories)

Stratified sampling by income categories

Custom feature engineering

Model comparison (Linear Regression, Decision Tree, Random Forest, SVR)

Hyperparameter tuning with GridSearchCV

Confidence interval evaluation

Installation
git clone https://github.com/yourusername/housing-price-prediction.git

cd housing-price-prediction

python -m venv venv

source venv/bin/activate (Windows: venv\Scripts\activate)

pip install -r requirements.txt

Usage
Run data processing: python src/data_processing.py
Train models: python src/model_training.py
Make predictions:
import joblib
model = joblib.load('models/tunned_Random_forest.pkl')
predictions = model.predict(new_data)

Project Structure
project/
├── data/
│ ├── raw/housing.csv
│ ├── processed/
│ └── interim/
├── models/
├── reports/figures/
├── src/
│ ├── data_processing.py
│ ├── model_training.py
│ └── custom_transformations.py
└── requirements.txt

Results
Best Model (Random Forest):

Test R²: 0.81

Test RMSE: $48,215

95% Confidence Interval: ± $1,850

Requirements
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0

License
MIT License - See LICENSE for details.

This version has:

All content in one continuous block

Minimal separation between sections

No isolated code blocks

Preserved all key information

Maintained proper Markdown formatting

Kept essential visual elements like badges
