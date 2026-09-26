# 🔮 ChurnGuard — Customer Churn Prediction

A simple **Machine Learning project** that predicts whether a telecom customer is likely to **CHURN or STAY**.

## 🚀 Project Overview

ChurnGuard takes customer information, cleans messy real-world data, trains a **Logistic Regression** model, and provides an instant churn prediction.

### ✨ What It Does

* 📂 Loads and explores the telecom dataset
* 🧹 Cleans missing, duplicate, and inconsistent data
* 🔢 Encodes categorical and target variables
* 🤖 Trains a Logistic Regression model
* 📊 Evaluates model performance
* 🔮 Predicts customer churn from user input

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* PyCharm

## 📁 Project Structure

```text
ChurnGuard/
│
├── churnguard_data.csv
├── task1_load_explore.py
├── task2_clean_data.py
├── task3_train_model.py
├── task4_predict.py
└── README.md
```

## ▶️ Run the Project

Install dependencies:

```bash
pip install pandas scikit-learn
```

Run the prediction script:

```bash
python task4_predict.py
```

Enter:

```text
Tenure
Monthly Charges
Total Charges
Senior Citizen
Contract Type
```

The model will return:

```text
Prediction: This customer is likely to CHURN.
```

or

```text
Prediction: This customer is likely to STAY.
```

## 🎯 Goal

**Turn customer data into actionable churn predictions.**

> 💡 Clean Data → Train Model → Predict Churn → Retain Customers
