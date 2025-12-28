# Customer Churn Prediction using Logistic Regression


#  Project Overview

This project predicts whether a customer is likely to churn (leave) or stay with a telecom service provider based on customer demographic and service usage data.

A Logistic Regression model is used to estimate churn probability, making the system interpretable, reliable, and suitable for business decision-making.

# Objectives

->Load and validate customer churn data

->Perform data preprocessing and encoding

->Apply feature scaling for Logistic Regression

->Train and evaluate a churn prediction model

->Predict churn probability for new customers using CLI

# Machine Learning Model

->Algorithm: Logistic Regression

->Library: Scikit-learn

->Problem Type: Binary Classification (Churn / No Churn)

->Output: Churn Probability

# Project Structure
customer-churn-logistic/
│
├── src/
│   ├── churn_training.py        # Model training script
│   └── churn_predict_cli.py     # Command-line prediction script
│
├── model/
│   ├── logistic_model.pkl       # Trained model
│   ├── feature_columns.pkl     # Feature list
│   └── scaler.pkl              # StandardScaler
│
├── data/
│   └── raw/
│       └── telco_churn.csv      # Dataset
│
├── requirements.txt
└── README.md

# Dataset Description

Dataset: Telco Customer Churn (Kaggle)

Feature	Description
gender	Male / Female
SeniorCitizen	Senior citizen (1/0)
Partner	Yes / No
Dependents	Yes / No
tenure	Number of months with company
PhoneService	Yes / No
InternetService	DSL / Fiber optic / No
MonthlyCharges	Monthly billing amount
TotalCharges	Total billing amount
Churn	Target variable (Yes / No)

# Installation & Setup
1️⃣ Clone Repository
git clone <your-github-repo-url>
cd customer-churn-logistic

2️⃣ Install Dependencies
pip install -r requirements.txt

# Model Training

Run the training script from the src folder:

python churn_training.py


This will:

->Preprocess the data

->Train the Logistic Regression model

->Evaluate Accuracy and F1-score

->Save the trained model, scaler, and feature list

# Prediction (Command-Line Interface)

Run:

python churn_predict_cli.py

Sample Input
Enter Gender (Male/Female): Female
Enter Senior Citizen (1 = Yes, 0 = No): 0
Enter Partner (Yes/No): Yes
Enter Dependents (Yes/No): No
Enter Tenure (months): 5
Enter Phone Service (Yes/No): Yes
Enter Internet Service (DSL/Fiber optic/No): Fiber optic
Enter Monthly Charges: 85.5
Enter Total Charges: 450.3

Sample Output
Churn Probability: 47.43 %
Prediction: Customer is likely to STAY


A probability threshold is used to control churn decisions.

# Evaluation Metrics

Accuracy

F1-Score

Confusion Matrix

Churn Probability

These metrics help assess model reliability for business use.

# Tools & Technologies

Python

Pandas

NumPy

Scikit-learn

Joblib

Machine Learning

Data Preprocessing

# Future Enhancements

Add contract & payment method features

Feature coefficient visualization

Streamlit / Flask web application

Hyperparameter tuning





# Author

Aman Kumar
Data Science & Machine Learning Project