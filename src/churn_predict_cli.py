import pandas as pd
import joblib

# Load saved model, features, and scaler
model = joblib.load(r"C:\Users\kumar\customer-churn-logistic\model\logistic_model.pkl")
feature_columns = joblib.load(r"C:\Users\kumar\customer-churn-logistic\model\feature_columns.pkl")
scaler = joblib.load(r"C:\Users\kumar\customer-churn-logistic\model\scaler.pkl")

print("---- CUSTOMER CHURN PREDICTION SYSTEM ----")

# Take user input
gender = input("Enter Gender (Male/Female): ")
SeniorCitizen = int(input("Enter Senior Citizen (1 = Yes, 0 = No): "))
Partner = input("Enter Partner (Yes/No): ")
Dependents = input("Enter Dependents (Yes/No): ")
tenure = int(input("Enter Tenure (months): "))
PhoneService = input("Enter Phone Service (Yes/No): ")
InternetService = input("Enter Internet Service (DSL/Fiber optic/No): ")
MonthlyCharges = float(input("Enter Monthly Charges: "))
TotalCharges = float(input("Enter Total Charges: "))

# Create input dictionary
new_customer = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "InternetService": InternetService,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Convert to DataFrame
df = pd.DataFrame([new_customer])

# Encode categorical variables
df = pd.get_dummies(df)

# Align with training features
df = df.reindex(columns=feature_columns, fill_value=0)

# Scale features
df_scaled = scaler.transform(df)

# Predict probability
probability = model.predict_proba(df_scaled)[0][1]

print("\nChurn Probability:", round(probability * 100, 2), "%")

# Decision threshold
if probability >= 0.45:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")
