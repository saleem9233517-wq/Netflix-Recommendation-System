import pandas as pd


# --------------------------------
# 1. Load Netflix Dataset
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")


# --------------------------------
# 2. Analyze Rating Categories
# --------------------------------

print("======================================")
print("Netflix Audience Rating Classification")
print("======================================")

print("\nDataset Shape:")
print(df.shape)

print("\nRating Categories:")
print(df["rating"].value_counts(dropna=False))

print("\nMissing Rating Values:")
print(df["rating"].isna().sum())

print("\nUnique Rating Categories:")
print(df["rating"].dropna().unique())

# --------------------------------
# 3. Select Relevant Features
# --------------------------------

df = df[
    [
        "type",
        "country",
        "release_year",
        "duration",
        "listed_in",
        "rating"
    ]
].copy()


# --------------------------------
# 4. Handle Missing Values
# --------------------------------

categorical_columns = [
    "type",
    "country",
    "duration",
    "listed_in"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")
    df[column] = df[column].astype(str)


# --------------------------------
# 5. Separate Features and Target
# --------------------------------

X = df.drop("rating", axis=1)
y = df["rating"]


# --------------------------------
# 6. Encode Categorical Features
# --------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# --------------------------------
# 7. Convert Features to Numeric
# --------------------------------

X_encoded = X_encoded.astype(float)


# --------------------------------
# 8. Display Prepared Dataset
# --------------------------------

print("\n======================================")
print("Training Dataset Prepared")
print("======================================")

print("\nFeature Shape:")
print(X_encoded.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Categories:")
print(y.unique())

print("\nFirst 5 Feature Records:")
print(X_encoded.head())

# --------------------------------
# 9. Split Training and Testing Data
# --------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\n======================================")
print("Training and Testing Data")
print("======================================")

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# --------------------------------
# 10. Train Classification Models
# --------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    solver="saga",
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)


# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)


print("\n======================================")
print("Multi-Class Models Trained")
print("======================================")

print("\nNumber of Rating Classes:", len(label_encoder.classes_))

print("\nLogistic Regression: Trained Successfully")
print("Random Forest: Trained Successfully")



# --------------------------------
# 11. Evaluate Models
# --------------------------------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
logistic_predictions = logistic_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

# Accuracy
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\n======================================")
print("Model Accuracy")
print("======================================")

print(
    "\nLogistic Regression Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest Accuracy:",
    round(random_forest_accuracy * 100, 2),
    "%"
)

# Logistic Regression Report
print("\n======================================")
print("Logistic Regression Classification Report")
print("======================================")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# Random Forest Report
print("\n======================================")
print("Random Forest Classification Report")
print("======================================")

print(
    classification_report(
        y_test,
        random_forest_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# Confusion Matrices
print("\n======================================")
print("Logistic Regression Confusion Matrix")
print("======================================")

print(confusion_matrix(y_test, logistic_predictions))

print("\n======================================")
print("Random Forest Confusion Matrix")
print("======================================")

print(confusion_matrix(y_test, random_forest_predictions))

# --------------------------------
# 12. Final Model Comparison
# --------------------------------

print("\n======================================")
print("Final Model Comparison")
print("======================================")

print(
    "\nLogistic Regression:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(random_forest_accuracy * 100, 2),
    "%"
)

accuracy_difference = (
    random_forest_accuracy - logistic_accuracy
) * 100

print(
    "\nAccuracy Difference:",
    round(accuracy_difference, 2),
    "%"
)
