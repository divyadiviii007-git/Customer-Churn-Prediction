# Customer Churn Prediction

## Synopsis
This project uses machine learning to predict whether a customer is likely to stop using a company's service. It analyzes customer tenure, monthly charges, service type, contract type, payment method, support, and internet service.

## Objectives
- Understand customer data.
- Analyze factors related to customer churn.
- Identify patterns in customer behavior.
- Build a classification model.
- Predict whether a customer may leave.
- Find important factors affecting churn.

## Tools
Python, Pandas, Matplotlib, Scikit-learn

## Dataset Features
- customer_id
- age
- tenure_months
- monthly_charges
- service_type
- payment_method
- contract_type
- tech_support
- internet_service
- churn

Target:
- 0 = Customer Stayed
- 1 = Customer Churned

## Data Preparation
- Loaded customer data using Pandas.
- Checked for missing values.
- Filled missing numerical values with median values.
- Filled missing categorical values with the most frequent value.
- Standardized numerical features.
- One-hot encoded categorical features.
- Split the data into training and testing sets.

## Machine Learning Algorithm
Logistic Regression is used for binary classification.

## Evaluation
The program calculates:
- Accuracy
- Classification report
- Confusion matrix
- Churn probability for a new customer

It also identifies the most influential model features and creates a churn-rate chart by contract type.

## How to Run
1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   python customer_churn_prediction.py

## Dataset Note
The included dataset is synthetic and designed for educational/classroom machine-learning practice. It should not be treated as real customer data.
