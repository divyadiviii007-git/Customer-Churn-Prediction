# Customer-Churn-Prediction
Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to stay or leave a service. Using Python, Pandas, and Scikit-learn, the project preprocesses customer data, trains a Logistic Regression model, evaluates its performance, identifies important churn factors, and predicts churn probability.
# Customer Churn Prediction

## Description

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to stay or leave a service. The project uses customer information to train a Logistic Regression model and provides churn predictions, churn probability, model evaluation, important churn factors, and visualization.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Logistic Regression

## Dataset

The project uses a CSV dataset named `customer_churn.csv`.

### Features Used

**Numerical Features:**

* Age
* Tenure Months
* Monthly Charges

**Categorical Features:**

* Service Type
* Payment Method
* Contract Type
* Tech Support
* Internet Service

The `customer_id` column is excluded from the model, while `churn` is used as the target variable.

## Project Workflow

1. Load the customer churn dataset.
2. Display the dataset and check for missing values.
3. Separate features and the target variable.
4. Identify numerical and categorical features.
5. Handle missing values.
6. Standardize numerical features.
7. Encode categorical features using One-Hot Encoding.
8. Split the dataset into 80% training and 20% testing data.
9. Train a Logistic Regression classification model.
10. Predict customer churn on test data.
11. Evaluate the model using accuracy, classification report, and confusion matrix.
12. Predict churn probability for a new customer.
13. Identify the most important factors affecting churn.
14. Visualize churn rate by contract type.

## Machine Learning Model

The project uses **Logistic Regression** for binary classification.

The model predicts two outcomes:

* `0` - Customer Stayed
* `1` - Customer Churned

The model also calculates the probability of a customer churning.

## Data Preprocessing

The project uses Scikit-learn pipelines for preprocessing.

### Numerical Data

* Missing values are replaced using the median.
* Features are standardized using `StandardScaler`.

### Categorical Data

* Missing values are replaced using the most frequent value.
* Categorical values are converted into numerical values using `OneHotEncoder`.

## Model Evaluation

The trained model is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

These metrics help measure how effectively the model predicts customer churn.

## New Customer Prediction

The project includes an example customer and predicts whether the customer is likely to churn or stay.

It also displays the customer's **churn probability**.

## Churn Factors

The Logistic Regression coefficients are analyzed to identify the top factors influencing customer churn. The factors are sorted according to their absolute importance.

## Visualization

The project creates a bar chart showing the churn rate for different contract types.

The generated visualization is saved as:

`churn_by_contract.png`

## Installation

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

1. Clone or download this repository.
2. Make sure `customer_churn.csv` and `customer_churn_prediction.py` are in the same folder.
3. Open the terminal in the project directory.
4. Run:

```bash
python customer_churn_prediction.py
```

## Project Structure

```text
Customer-Churn-Prediction/
│
├── customer_churn_prediction.py
├── customer_churn.csv
├── churn_by_contract.png
└── README.md
```

## Real-World Application

Customer churn prediction can help businesses identify customers who may leave their service. Businesses can use these predictions to understand customer behavior and take suitable retention measures.

## Future Improvements

* Compare Logistic Regression with other machine learning algorithms.
* Improve model performance using hyperparameter tuning.
* Add more customer features.
* Create an interactive dashboard.
* Deploy the model as a web application.

## Conclusion

This project demonstrates the use of machine learning to predict customer churn. It covers data preprocessing, model training, evaluation, prediction, feature importance analysis, and visualization using Python and Scikit-learn.
