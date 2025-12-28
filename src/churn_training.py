import pandas as pd

data = pd.read_csv(r"C:\Users\kumar\customer-churn-logistic\data\raw\telco_churn.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
data.drop("customerID", axis=1, inplace=True)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["Churn"] = le.fit_transform(data["Churn"])  # Yes=1, No=0
data = pd.get_dummies(data, drop_first=True)
X = data.drop("Churn", axis=1)
y = data["Churn"]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, r"C:\Users\kumar\customer-churn-logistic\model\logistic_model.pkl")
joblib.dump(X.columns.tolist(), r"C:\Users\kumar\customer-churn-logistic\model\feature_columns.pkl")
joblib.dump(scaler, r"C:\Users\kumar\customer-churn-logistic\model\scaler.pkl")

print("Model saved successfully!")
