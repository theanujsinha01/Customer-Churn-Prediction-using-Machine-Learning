
# Customer Churn Prediction using Machine Learning 🚀

**Predicting which customers are likely to leave a telecom service using machine learning.**

---

### 💼 Problem Statement

Telecom companies face huge losses when customers leave (churn).  
Retaining old customers is cheaper than getting new ones.  
This app helps predict which customers are at risk so the company can act early.

---

### 📊 Dataset Info

- **Source:** Telco Customer Churn Dataset (Kaggle)  
- **Total Customers:** 7,043  
- **Target Variable:** `Churn` (Yes/No)  
- **Features:**  
  - Demographics (Gender, Senior Citizen, etc.)  
  - Services used (Internet, Streaming, etc.)  
  - Account details (Tenure, Monthly Charges, Contract Type, etc.)  
- **Class Split:**  
  - 26.5% Churned  
  - 73.5% Retained

---

### 🔧 What This App Does

- Takes customer info as input (via Streamlit form)
- Applies same **Label Encoding** used during model training
- Uses a pre-trained **Random Forest model**
- Shows real-time prediction: Will customer churn or not?

---

### 🧠 Machine Learning Details

- Used **Random Forest Classifier**
- Handled class imbalance using **SMOTE**
- Encoded all categorical columns with **LabelEncoder**
- Model trained and saved using `joblib`
- Input features reordered to match training data
- App deployed using **Streamlit**

---

### 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Streamlit  
- Joblib

---

### ▶️ How to Run

```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction-using-Machine-Learning.git
cd Customer-Churn-Prediction-using-Machine-Learning
pip install -r requirements.txt
streamlit run app.py

