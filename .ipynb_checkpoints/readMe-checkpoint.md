# 🍷 Wine Quality Prediction (Supervised Classification)

This project predicts whether a wine is **Good** or **Bad** based on its chemical properties using **Machine Learning**.  
It includes model training, evaluation, and deployment using **Streamlit**.

---

## 📌 Problem Statement

Wine quality depends on multiple chemical characteristics.  
The goal of this project is to classify wine quality into:

- **Good Wine (1)** → Quality > 5  
- **Bad Wine (0)** → Quality ≤ 5  

This is a **Supervised Binary Classification** problem.

---

## 📊 Dataset

- Source: Kaggle – Wine Quality Dataset
- File: `WineQT.csv`

### Removed Columns (Not Affecting Quality)
- Id  
- fixed acidity  
- citric acid  
- free sulfur dioxide  
- residual sugar  
- chlorides  
- quality (used only to create target)

---

## 🔑 Features Used

- volatile acidity  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Feature Selection  
2. Exploratory Data Analysis (EDA)  
3. Feature Scaling (StandardScaler)  
4. Model Training:
   - Logistic Regression
   - Decision Tree
   - **Random Forest (Final Model)**
5. Model Evaluation:
   - Accuracy
   - Confusion Matrix
   - ROC-AUC
6. Hyperparameter Tuning (GridSearchCV)
7. Deployment using Streamlit

---

## 🚀 Streamlit Application

The Streamlit app allows users to:
- Enter wine chemical values
- Predict wine quality
- View prediction confidence
- Preview dataset (20 rows)

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
