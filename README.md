# Fraud Detection in Online Transactions

## Overview

This project detects fraudulent credit card transactions using Machine Learning.
It handles imbalanced data and provides real-time prediction using a Flask API.

## Tech Stack

Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Flask

## Dataset

Kaggle Credit Card Fraud Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains 284,807 transactions with very few fraud cases.

## Project Files

fraud_detection.py – model training
app.py – Flask API
model.pkl – trained model
requirements.txt – dependencies

## How it Works

The data is preprocessed by scaling the Amount column and removing unnecessary features.
SMOTE is used to handle class imbalance.
Two models are trained: Logistic Regression and Random Forest.
Random Forest is used as the final model.
The model is evaluated using precision, recall, and ROC-AUC.

## How to Run

Install dependencies
pip install -r requirements.txt

Run model training
python fraud_detection.py

Run the API
python app.py

## API Usage

Open browser
http://127.0.0.1:5000/

For prediction use POST request to
http://127.0.0.1:5000/predict

Example input
{
"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

Output will show whether the transaction is fraud and its probability.

## Key Points

Handles highly imbalanced data using SMOTE
Focuses on recall for better fraud detection
Provides real-time prediction using Flask

## Author

Mohammed Abzel F
