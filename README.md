🏠 House Price Prediction using Machine Learning

This project builds and compares multiple regression models to predict house prices using the California Housing dataset. It covers data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and real-time prediction.

📌 Overview

The goal of this project is to predict median house values using numerical housing features.
The workflow includes:

Data cleaning & preprocessing

Scaling numerical features

Training multiple ML models

Evaluating models using RMSE and R² Score

Hyperparameter tuning using GridSearchCV

Saving and loading the best-performing model

Predicting prices for new house samples

🛠️ Tech Stack

Python

Scikit-Learn

XGBoost

Pandas

NumPy

Matplotlib

Joblib

📥 Dataset

This project uses the California Housing dataset from Scikit-Learn:

from sklearn.datasets import fetch_california_housing


It contains housing-related numerical features such as:

Median income

House age

Average rooms

Average bedrooms

Population

Latitude & Longitude

Target variable:
MedHouseVal → renamed to target

🚀 Project Pipeline
1️⃣ Load & Explore Data

Loaded California housing data using fetch_california_housing

Checked shape, summary statistics, distributions, correlations

2️⃣ Preprocessing

Standardized all numerical features using StandardScaler

Split data into train/test sets (80/20)

3️⃣ Models Trained

Linear Regression

Random Forest Regressor

XGBoost Regressor

4️⃣ Model Evaluation

Metrics used:

RMSE (Root Mean Squared Error)

R² Score

5️⃣ Hyperparameter Tuning

Used GridSearchCV with cross-validation to optimize the best model.

6️⃣ Saving the Best Model

Saved the optimized model using:

joblib.dump(best_model, "house_price_best_model.pkl")

7️⃣ Prediction on New Data

Used the trained model to predict house price for new samples:

pred = best_model.predict(sample)

📊 Results

Model comparison (example):

Model	RMSE ↓	R² ↑
Linear Regression	0.73	0.60
Random Forest	0.52	0.80
XGBoost	0.46	0.84

👉 XGBoost performed the best, and was chosen as the final model.

🧪 Sample Prediction
sample = X_test.iloc[[1]]
pred = best_model.predict(sample)

print("Predicted price:", pred[0])
print("Actual price:", y_test.iloc[1])
<img width="1064" height="228" alt="image" src="https://github.com/user-attachments/assets/d2de01da-c13d-4edf-b8fe-4c2c2bd756d4" />
