# BLEVE Target Pressure Prediction

A machine learning project for predicting target pressure in Boiling Liquid Expanding Vapor Explosion (BLEVE) scenarios using various regression algorithms.

## Project Overview

This project implements and compares multiple regression models to predict target pressure in BLEVE scenarios based on tank properties, obstacle configurations, sensor positions, and liquid characteristics.

## Dataset

- **Training Data**: 10,050 records with 30 features
- **Test Data**: Unlabeled dataset for final predictions
- **Target Variable**: Target Pressure (bar)

### Key Features
- Tank properties (dimensions, failure pressure, liquid ratio)
- Obstacle characteristics (distance, dimensions, angle)
- Sensor positioning (x, y, z coordinates, side)
- Liquid properties (temperature, critical pressure/temperature)
- Engineered features (vapor volume, tank area, sensor distances)

## Data Preprocessing

### Feature Engineering
- **Tank Volume**: `length × width × height`
- **Vapor Volume**: `(1 - liquid_ratio) × tank_volume`
- **Vapor Force**: `tank_failure_pressure × (vapor_height × tank_length)`
- **Sensor Distance to Tank**: Euclidean distance from sensor to tank center
- **Tank Area**: Total surface area calculation
- **Sensor Distance to BLEVE**: Combined distance calculations

### Data Cleaning
- Outlier removal using Z-score (threshold = 3)
- Missing value handling
- Duplicate record removal
- Label encoding for categorical variables ('Status')
- MinMax scaling for numerical features

## Models Implemented

### 1. Linear Regression
- **R² Score**: Basic performance baseline
- Selected top 10 most important features
- Used for initial feature importance analysis

### 2. Gradient Boosting Regression
- **Hyperparameters**: 
  - `learning_rate=0.1`
  - `max_depth=7` 
  - `n_estimators=1000`
  - `subsample=0.75`
- **Performance**: R² = 0.999877, MAPE = 0.01881

### 3. XGBoost Regression
- **Hyperparameters**:
  - `eta=0.1`
  - `max_depth=7`
  - `n_estimators=300`
  - `subsample=0.5`
- **Performance**: High accuracy with feature importance ranking

### 4. Ensemble Model (Voting Regressor)
- Combines XGBoost and Gradient Boosting
- **Performance**: R² = 0.999886, MAPE = 0.01747
- Best overall performance

### 5. Support Vector Regression (SVR)
- RBF kernel implementation
- Hyperparameter tuning for C, kernel, and degree

### 6. Neural Network
- **Architecture**: 
  - Input layer: 29 features
  - Hidden layers: [256, 128, 64, 32, 16] neurons
  - Activation: PReLU
  - Dropout: 0.2
  - Gaussian noise: 0.05
- **Optimizer**: Nadam with custom parameters
- **Tuned Version**: Keras Tuner optimization with 156-52-29-23-12 architecture

## Feature Selection Methods

- **Mutual Information Regression**: Information-theoretic approach
- **Fisher Score**: Statistical feature ranking
- **Extra Trees Feature Importance**: Tree-based importance
- **Correlation Analysis**: Heatmap visualization

## Results Summary

| Model | R² Score | MAPE | MAE | RMSE |
|-------|----------|------|-----|------|
| Gradient Boosting | 0.999877 | 0.01881 | 0.00277 | 0.00341 |
| XGBoost | - | - | - | - |
| Ensemble | 0.999886 | 0.01747 | 0.00258 | 0.00329 |
| Neural Network (Tuned) | - | - | - | - |

## Files Generated

- `gbr_prediction.csv`: Gradient Boosting predictions
- `xgb_prediction.csv`: XGBoost predictions  
- `ensembled_regressor_prediction.csv`: Ensemble model predictions
- `nn_prediction.csv`: Neural Network predictions
- `nn_tuned_prediction.csv`: Tuned Neural Network predictions

## Installation

```bash
pip install scikit-learn==1.3.2
pip install skfeature-chappers
pip install mlxtend 
pip install pydataset
pip install xgboost
pip install keras-tuner 
pip install scikeras
```

## Usage

1. Load training data: `train.csv`
2. Run feature engineering and preprocessing
3. Train multiple regression models
4. Generate predictions on test data
5. Export results to CSV files

## Key Insights

- Engineered features significantly improved model performance
- Ensemble methods provided the best predictive accuracy
- Neural networks showed competitive performance with proper tuning
- Feature importance analysis revealed vapor force and sensor distances as critical predictors

## Future Work

- Implement additional ensemble techniques
- Explore deep learning architectures
- Feature selection optimization
- Cross-validation improvements