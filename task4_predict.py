import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Load the dataset
df = pd.read_csv("churnguard_data.csv")

# -----------------------------
# Task 2 - Cleaning the dataset
# -----------------------------

# Drop customerID
df = df.drop("customerID", axis=1)

# Remove duplicate rows
df = df.drop_duplicates()

# Strip whitespace
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()

# Standardise casing
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()

# Fix Contract variations
df["Contract"] = df["Contract"].str.strip()

contract_mapping = {
    "Monthly": "Month-to-month",
    "monthly": "Month-to-month",
    "month to month": "Month-to-month",
    "Month to month": "Month-to-month",
    "month-to-month": "Month-to-month",
    "Month-to-month": "Month-to-month",

    "1 year": "One year",
    "one year": "One year",
    "One year": "One year",

    "2 year": "Two year",
    "two year": "Two year",
    "Two year": "Two year"
}

df["Contract"] = df["Contract"].replace(contract_mapping)

# Fix InternetService variations
df["InternetService"] = df["InternetService"].str.strip()

internet_mapping = {
    "dsl": "DSL",
    "DSL": "DSL",

    "Fibre optic": "Fiber optic",
    "Fiber optic": "Fiber optic",
    "FiberOptic": "Fiber optic",
    "fiber optic": "Fiber optic",

    "None": "No",
    "none": "No",
    "No": "No",
    "no": "No"
}

df["InternetService"] = df["InternetService"].replace(
    internet_mapping
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where tenure is zero or negative
df = df[df["tenure"] > 0]

# Remove rows where MonthlyCharges is less than 10
# or greater than 200
df = df[
    (df["MonthlyCharges"] >= 10) &
    (df["MonthlyCharges"] <= 200)
]

# Fill missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].mean()
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)

df["tenure"] = df["tenure"].fillna(
    round(df["tenure"].median())
).astype(int)

# -----------------------------
# Task 4 - Prediction Model
# -----------------------------

# Encode Churn
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Encode Contract
contract_numeric = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

df["Contract"] = df["Contract"].map(contract_numeric)

# Remove any rows where Contract could not be mapped
df = df.dropna(subset=["Contract"])

# Convert Contract to integer
df["Contract"] = df["Contract"].astype(int)

# Select the five required features
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract"
]

X = df[features]
y = df["Churn"]

# Make sure there are no missing values
X = X.fillna(X.mean())

# Train Logistic Regression on full cleaned dataset
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------
# Collect user input
# -----------------------------

tenure = int(input("Enter tenure (months): "))

monthly_charges = float(
    input("Enter Monthly Charges: ")
)

total_charges = float(
    input("Enter Total Charges: ")
)

senior_citizen = int(
    input("Senior Citizen? (1 = Yes, 0 = No): ")
)

contract = int(
    input(
        "Contract type (0 = Month-to-month, "
        "1 = One year, 2 = Two year): "
    )
)

# Create DataFrame for prediction
customer_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": senior_citizen,
    "Contract": contract
}])

# Predict churn
prediction = model.predict(customer_data)[0]

# Print result
if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")