import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("churnguard_data.csv")

# -----------------------------
# Task 2 - Cleaning the dataset
# -----------------------------

# 2. Drop customerID
df = df.drop("customerID", axis=1)

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Strip whitespace
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

# 10. Remove rows where MonthlyCharges is less than 10
#     or greater than 200
df = df[
    (df["MonthlyCharges"] >= 10) &
    (df["MonthlyCharges"] <= 200)
]

# 11. Fill missing values
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
# Task 3 - Train the model
# -----------------------------

# 12. Encode the target column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 13. Encode categorical columns using get_dummies
categorical_columns = [
    "gender",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# 14. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 15. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 16. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 17. Make predictions
y_pred = model.predict(X_test)

# 18. Print accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:")
print(accuracy)

# 19. Print classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stay", "Churn"]
    )
)