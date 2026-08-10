# 💰 Loan Approval Predictor

An end-to-end **Machine Learning based Loan Approval Prediction System** that predicts whether a loan application is likely to be approved or rejected based on applicant financial and credit-related information.

The project includes **data preprocessing, exploratory data analysis, handling imbalanced data, multiple machine learning models, model evaluation, hyperparameter tuning, feature importance, model serialization, and a Streamlit web application**.

---

## 🚀 Project Overview

Loan approval decisions depend on several factors such as:

* Applicant income
* Age
* Employment length
* Home ownership
* Loan amount
* Loan interest rate
* Loan purpose
* Loan grade
* Previous credit default history
* Credit history length
* Loan-to-income ratio

This project uses these features to build a classification model for predicting the **loan status**.

### 🎯 Objective

The main objective of this project is to build a machine learning system that can:

1. Analyze loan applicant data
2. Preprocess and clean the dataset
3. Handle class imbalance
4. Train multiple classification models
5. Compare model performance
6. Select the best-performing model
7. Tune model hyperparameters
8. Evaluate predictions
9. Identify important features
10. Provide predictions through a Streamlit application

---

## 📊 Dataset

The project uses the **Credit Risk Dataset** containing **32,581 records and 12 columns**.

### Dataset Features

| Feature                      | Description                           |
| ---------------------------- | ------------------------------------- |
| `person_age`                 | Applicant's age                       |
| `person_income`              | Applicant's annual income             |
| `person_home_ownership`      | Home ownership status                 |
| `person_emp_length`          | Employment length                     |
| `loan_intent`                | Purpose of the loan                   |
| `loan_grade`                 | Loan grade                            |
| `loan_amnt`                  | Loan amount                           |
| `loan_int_rate`              | Loan interest rate                    |
| `loan_status`                | Target variable                       |
| `loan_percent_income`        | Loan amount as a percentage of income |
| `cb_person_default_on_file`  | Previous credit default indicator     |
| `cb_person_cred_hist_length` | Length of credit history              |

### Target Variable

```text
loan_status
```

The model predicts the loan status based on the available applicant and loan-related features.

---

## 🧠 Machine Learning Workflow

The complete workflow followed in this project is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Feature / Target Separation
   ↓
SMOTE
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Saving
   ↓
Streamlit Application
```

---

## 🧹 Data Preprocessing

The dataset is processed before training the machine learning models.

### Missing Values

Missing values are checked and handled using forward filling.

```python
df.fillna(method='ffill', inplace=True)
```

### Duplicate Removal

Duplicate records are removed:

```python
df.drop_duplicates(inplace=True)
```

### Categorical Encoding

Categorical features are converted into numerical values using `LabelEncoder`.

Examples include:

* Home ownership
* Loan intent
* Loan grade
* Previous default status

---

## ⚖️ Handling Imbalanced Data

Loan datasets can contain an unequal number of approved and rejected loan records.

To address class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** is used.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

This helps the models learn from both classes more effectively.

---

## 🤖 Machine Learning Models

The notebook trains and compares multiple classification algorithms.

### 1. Logistic Regression

A linear classification algorithm used as a baseline model.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

### 3. XGBoost

A powerful gradient boosting algorithm used for classification problems.

The models are compared using **Accuracy Score**.

---

## 🏆 Model Selection

The model with the highest accuracy is selected as the best-performing model.

```python
best_model_name = max(results, key=results.get)
```

The project then performs additional hyperparameter tuning using `GridSearchCV`.

For Random Forest, parameters such as:

```text
n_estimators
max_depth
```

are optimized.

---

## 📈 Model Evaluation

The selected model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

Example evaluation:

```python
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

A confusion matrix is also generated to understand correct and incorrect classifications.

---

## 🔍 Feature Importance

The project analyzes which features have the greatest influence on the model's predictions.

Feature importance visualization helps understand factors contributing to loan approval predictions.

```python
importances = best_model.feature_importances_
```

This makes the model more interpretable and useful for understanding applicant risk factors.

---

## 💾 Model Saving

The trained model and scaler are saved using `joblib`.

```python
joblib.dump(best_model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

This allows the trained model to be reused without retraining every time.

---

# 🌐 Streamlit Application

The project also includes a **Streamlit-based web application**.

The application provides an interactive interface for working with the loan prediction system.

## 🔐 Authentication

The application supports:

* User registration
* Login
* User/Admin roles
* Password hashing
* Session management
* Logout
* JWT-based session token

---

## 📂 Dataset Upload

Users can upload a CSV dataset directly through the application.

The application allows the user to:

* Upload dataset
* Preview the dataset
* Select target column
* Define prediction features

---

## 🧪 Model Training

The application can train multiple models including:

* Random Forest
* Logistic Regression
* XGBoost (when available)

The application displays model accuracy and selects the best-performing model.

It also generates a confusion matrix for evaluation.

---

## 📊 Dashboard

The dashboard provides data visualization using **Plotly**.

Users can select dataset columns and visualize their distributions.

---

## 🔮 Loan Prediction

After training the model, users can enter feature values and generate a prediction.

The application displays:

```text
Prediction: 0 / 1
```

The prediction can also be saved as a CSV report.

---

## 🧠 Model Explainability

The application includes an explainability section with:

### Feature Importance

Shows the importance of different features used by the model.

### SHAP Explainability

If SHAP is available, the application generates SHAP-based explanations to understand model behavior.

If SHAP is unavailable, the application provides a fallback visualization.

---

## 👨‍💼 Admin Panel

Administrators can access the Admin Panel to:

* View registered users
* Select users
* Delete users

This provides basic user-management functionality.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Explainable AI

* SHAP

### Web Application

* Streamlit

### Model Persistence

* Joblib

### Authentication

* Hashlib
* JWT

---

# 📁 Project Structure

```text
Loan-Approval-Predictor/
│
├── Loan Approval Predictor.ipynb
├── credit_risk_dataset.csv
├── app.py
├── loan_model.pkl
├── scaler.pkl
├── users.csv
├── prediction.csv
└── README.md
```

> `loan_model.pkl`, `scaler.pkl`, `users.csv`, and `prediction.csv` are generated/used during application execution as applicable.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Loan-Approval-Predictor.git
```

Move into the project directory:

```bash
cd Loan-Approval-Predictor
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib streamlit plotly shap pyjwt
```

---

# ▶️ Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Loan Approval Predictor.ipynb
```

Run the cells sequentially.

---

# 🌐 Run the Streamlit Application

After creating/placing `app.py` in the project directory:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📌 Key Features

✅ Loan approval prediction
✅ Data cleaning and preprocessing
✅ Exploratory Data Analysis
✅ Categorical encoding
✅ SMOTE-based imbalance handling
✅ Multiple ML models
✅ Logistic Regression
✅ Random Forest
✅ XGBoost
✅ Model comparison
✅ Hyperparameter tuning
✅ Confusion matrix
✅ Classification report
✅ Feature importance
✅ SHAP explainability
✅ Model serialization
✅ CSV prediction report
✅ Streamlit web application
✅ User registration/login
✅ Admin panel
✅ Role-based access

---

# 🎓 Project Use Cases

This project can be used as a foundation for:

* Credit risk analysis
* Loan approval assistance
* Financial risk prediction
* Banking analytics
* Applicant screening
* Machine learning classification
* Explainable AI applications

---

# 🔮 Future Improvements

The system can be further enhanced with:

* Advanced credit risk scoring
* Probability-based approval prediction
* Better categorical preprocessing using One-Hot Encoding
* Cross-validation based model comparison
* ROC-AUC and PR-AUC evaluation
* Advanced hyperparameter optimization
* Interactive applicant risk dashboard
* SHAP-based individual prediction explanations
* Database integration
* Cloud deployment
* REST API integration
* Automated model retraining
* Advanced role-based access control
* Real-time prediction monitoring

