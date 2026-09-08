# Customer Churn Prediction
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("customer_churn.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Features and target
X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]

# Identify numerical and categorical columns
numeric_features = ["age", "tenure_months", "monthly_charges"]
categorical_features = [
    "service_type", "payment_method", "contract_type",
    "tech_support", "internet_service"
]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Stayed", "Churned"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new customer
new_customer = pd.DataFrame([{
    "age": 28,
    "tenure_months": 5,
    "monthly_charges": 85,
    "service_type": "Basic",
    "payment_method": "Electronic Check",
    "contract_type": "Month-to-month",
    "tech_support": "No",
    "internet_service": "Fiber"
}])

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

print("\nExample Customer Prediction:",
      "Likely to Churn" if prediction == 1 else "Likely to Stay")
print(f"Churn Probability: {probability * 100:.2f}%")

# Find important factors from the trained logistic regression model
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_importance": abs(coefficients)
}).sort_values("absolute_importance", ascending=False)

print("\nTop Factors Affecting Churn:")
print(importance.head(10)[["feature", "coefficient"]])

# Visualization: churn by contract type
churn_by_contract = data.groupby("contract_type", dropna=False)["churn"].mean() * 100

plt.figure(figsize=(8, 5))
churn_by_contract.plot(kind="bar")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.title("Customer Churn by Contract Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("churn_by_contract.png", dpi=150)
plt.show()
