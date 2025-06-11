# California Housing Price Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning project to predict median housing prices in California districts using various demographic and geographic features.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Comprehensive EDA**: Visualizations of feature distributions, geographical patterns, and correlations
- **Stratified Sampling**: Representative test sets based on income categories
- **Feature Engineering**:
  - Custom attribute addition (`AttributesAdder`)
  - One-Hot Encoding for categorical features
  - Median imputation for missing values
- **Model Comparison**: Linear Regression, Decision Tree, Random Forest, and SVR
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Robust Evaluation**: RMSE with confidence intervals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/california-housing-prediction.git
cd california-housing-prediction
