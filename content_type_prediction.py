import pandas as pd


# --------------------------------
# 1. Load Netflix Dataset
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")


# --------------------------------
# 2. Select Relevant Features
# --------------------------------

df = df[
    [
        "type",
        "director",
        "country",
        "release_year",
        "rating",
        "duration",
        "listed_in"
    ]
].copy()


# --------------------------------
# 3. Display Dataset Information
# --------------------------------

print("======================================")
print("Netflix Content Type Prediction")
print("======================================")

print("\nDataset Shape:")
print(df.shape)

print("\nSelected Features:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["type"].value_counts())

print("\nFirst 5 Records:")
print(df.head())

# --------------------------------
# 4. Handle Missing Values
# --------------------------------

categorical_columns = [
    "director",
    "country",
    "rating",
    "duration",
    "listed_in"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")
    df[column] = df[column].astype(str)


# --------------------------------
# 5. Encode Target Variable
# --------------------------------

df["type"] = df["type"].map({
    "Movie": 0,
    "TV Show": 1
})


# --------------------------------
# 6. One-Hot Encode Categorical Features
# --------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)


# --------------------------------
# 7. Display Encoded Dataset
# --------------------------------

print("\n======================================")
print("Encoded Dataset")
print("======================================")

print("\nEncoded Dataset Shape:")
print(df_encoded.shape)

print("\nEncoded Features:")
print(df_encoded.columns.tolist())

print("\nFirst 5 Encoded Records:")
print(df_encoded.head())
# --------------------------------
# 8. Prepare Features and Target
# --------------------------------

X = df_encoded.drop("type", axis=1)
y = df_encoded["type"]


# --------------------------------
# 9. Convert Data to Numeric Format
# --------------------------------

X = X.astype(float)
y = y.astype(int)


# --------------------------------
# 10. Split Dataset
# --------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n======================================")
print("Training and Testing Data")
print("======================================")

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# --------------------------------
# 11. Train Classification Models
# --------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train,
    y_train
)


# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)


print("\n======================================")
print("Classification Models Trained")
print("======================================")

print("\nLogistic Regression: Trained Successfully")
print("Random Forest: Trained Successfully")\


# --------------------------------
# 12. Evaluate Model Performance
# --------------------------------

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Make predictions
logistic_predictions = logistic_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)


# --------------------------------
# 13. Logistic Regression Evaluation
# --------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\n======================================")
print("Logistic Regression Performance")
print("======================================")

print(
    f"\nAccuracy: "
    f"{logistic_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Movie", "TV Show"]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)


# --------------------------------
# 14. Random Forest Evaluation
# --------------------------------

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\n======================================")
print("Random Forest Performance")
print("======================================")

print(
    f"\nAccuracy: "
    f"{random_forest_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        random_forest_predictions,
        target_names=["Movie", "TV Show"]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        random_forest_predictions
    )
)


# --------------------------------
# 15. Compare Model Accuracy
# --------------------------------

print("\n======================================")
print("Model Accuracy Comparison")
print("======================================")

print(
    f"\nLogistic Regression Accuracy: "
    f"{logistic_accuracy:.2%}"
)

print(
    f"Random Forest Accuracy: "
    f"{random_forest_accuracy:.2%}"
)

accuracy_difference = abs(
    logistic_accuracy - random_forest_accuracy
)

print(
    f"\nAccuracy Difference: "
    f"{accuracy_difference:.2%}"
)