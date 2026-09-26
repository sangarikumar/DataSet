import pandas as pd

# 1. Load the dataset
df = pd.read_csv("churnguard_data.csv")

# 2. Drop the customerID column
df = df.drop("customerID", axis=1)

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Strip whitespace from gender and PaymentMethod
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()

# 5. Standardise casing
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()

# 6. Fix Contract variations
contract_mapping = {
    "Monthly": "Month-to-month",
    "month to month": "Month-to-month",
    "Month to month": "Month-to-month",
    "Month-to-month": "Month-to-month",
    "1 year": "One year",
    "One year": "One year",
    "one year": "One year",
    "2 year": "Two year",
    "Two year": "Two year",
    "two year": "Two year"
}

df["Contract"] = (
    df["Contract"]
    .str.strip()
    .replace(contract_mapping)
)

# 7. Fix InternetService variations
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

df["InternetService"] = (
    df["InternetService"]
    .str.strip()
    .replace(internet_mapping)
)

# 8. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 9. Remove rows where tenure is zero or negative
df = df[df["tenure"] > 0]

# 10. Remove rows where MonthlyCharges is less than 10 or greater than 200
df = df[
    (df["MonthlyCharges"] >= 10) &
    (df["MonthlyCharges"] <= 200)
]

# 11. Fill missing values

# MonthlyCharges -> column mean
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].mean()
)

# TotalCharges -> column mean
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)

# tenure -> column median with integer rounding
df["tenure"] = df["tenure"].fillna(
    round(df["tenure"].median())
).astype(int)

# 12. Print the shape of the cleaned DataFrame
print("Shape of the cleaned DataFrame:")
print(df.shape)

# 13. Print missing value counts
print("\nMissing values after cleaning:")
print(df.isnull().sum())