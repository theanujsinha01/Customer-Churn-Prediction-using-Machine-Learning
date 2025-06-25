
# Customer Churn Prediction using Machine Learning 🚀

This project predicts customer churn (i.e., whether a customer will leave the company or not) based on customer usage behavior and services. It uses machine learning models trained on the popular Telco Customer Churn dataset.

## 📂 Project Files

| File/Folder                  | Description                                      |
|-----------------------------|--------------------------------------------------|
| `main.ipynb`                | Main Jupyter Notebook with all code              |
| `random_forest_model.pkl`   | Trained Random Forest model                      |
| `label_encoders.pkl`        | Encoders used to preprocess categorical data     |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original dataset                         |

---

## 🧠 Problem Statement

The goal is to build a machine learning model to:
- Predict whether a customer will **churn (leave)** or **stay**.
- Use features like gender, services subscribed, tenure, charges, etc.

---

## 📊 Dataset Info

- Source: IBM Sample Dataset
- Records: 7043
- Features: 19 input features + target `Churn` (Yes/No)

---

## ✅ Workflow Summary

1. **Data Preprocessing**
   - Missing value handling
   - Label Encoding (`label_encoders.pkl`)
   - Feature selection

2. **Model Training**
   - SMOTE applied to balance classes
   - Random Forest used as final model
   - Cross-validation applied for accuracy check

3. **Model Saving**
   - `random_forest_model.pkl` for model
   - `label_encoders.pkl` for encoders

4. **Prediction Pipeline**
   - Load model + encoders
   - Take new customer input
   - Predict churn

---

## 📦 How to Use

### 🔧 Step 1: Clone this repo

```bash
git clone https://github.com/yourusername/customer-churn-predictor.git
cd customer-churn-predictor
