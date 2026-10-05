import pandas as pd

# --------------------------------
# Netflix Content Success Analytics Engine
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")

print("======================================")
print("Netflix Content Success Analytics")
print("======================================")

print("\nOriginal Dataset Shape:")
print(df.shape)

# --------------------------------
# 1. Prepare Date Features
# --------------------------------

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

df["added_year"] = df["date_added"].dt.year
df["added_month"] = df["date_added"].dt.month

# --------------------------------
# 2. Create Content Age Feature
# --------------------------------

reference_year = int(df["release_year"].max())

df["content_age"] = (
    reference_year - df["release_year"]
)

# --------------------------------
# 3. Create Genre Count
# --------------------------------

df["genre_count"] = (
    df["listed_in"]
    .fillna("Unknown")
    .astype(str)
    .str.split(",")
    .apply(len)
)

# --------------------------------
# 4. Create Country Count
# --------------------------------

df["country_count"] = (
    df["country"]
    .fillna("Unknown")
    .astype(str)
    .str.split(",")
    .apply(len)
)

# --------------------------------
# 5. Create Duration Feature
# --------------------------------

df["duration_value"] = (
    df["duration"]
    .fillna("0")
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .fillna(0)
    .astype(float)
)

# --------------------------------
# 6. Create Director Availability
# --------------------------------

df["has_director"] = (
    df["director"]
    .notna()
    .astype(int)
)

# --------------------------------
# 7. Create Content Type Feature
# --------------------------------

df["is_movie"] = (
    df["type"] == "Movie"
).astype(int)

# --------------------------------
# 8. Select Engineered Features
# --------------------------------

engineered_features = [
    "release_year",
    "content_age",
    "added_year",
    "added_month",
    "genre_count",
    "country_count",
    "duration_value",
    "has_director",
    "is_movie"
]

analytics_df = df[
    engineered_features
].copy()

# --------------------------------
# 9. Handle Missing Values
# --------------------------------

analytics_df = analytics_df.fillna(0)

# --------------------------------
# 10. Display Engineered Features
# --------------------------------

print("\n======================================")
print("Advanced Features Created")
print("======================================")

print("\nFeature Shape:")
print(analytics_df.shape)

print("\nReference Year:")
print(reference_year)

print("\nEngineered Features:")
print(
    analytics_df.columns.tolist()
)

print("\nFirst 5 Records:")
print(
    analytics_df.head()
)

print("\nFeature Statistics:")
print(
    analytics_df.describe()
)

# --------------------------------
# 11. Create Content Success Proxy
# --------------------------------

print("\n======================================")
print("Creating Content Success Proxy")
print("======================================")

# Genre popularity
genre_frequency = (
    df["listed_in"]
    .fillna("Unknown")
    .value_counts()
)

df["genre_popularity"] = (
    df["listed_in"]
    .fillna("Unknown")
    .map(genre_frequency)
)

# Country popularity
country_frequency = (
    df["country"]
    .fillna("Unknown")
    .value_counts()
)

df["country_popularity"] = (
    df["country"]
    .fillna("Unknown")
    .map(country_frequency)
)

# Rating popularity
rating_frequency = (
    df["rating"]
    .fillna("Unknown")
    .value_counts()
)

df["rating_popularity"] = (
    df["rating"]
    .fillna("Unknown")
    .map(rating_frequency)
)

# Combined success proxy
df["success_score"] = (
    df["genre_popularity"]
    + df["country_popularity"]
    + df["rating_popularity"]
)

# Create three success categories
success_rank = df["success_score"].rank(
    method="first"
)

df["success_category"] = pd.qcut(
    success_rank,
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

print("\nSuccess Category Distribution:")
print(
    df["success_category"].value_counts()
)

print("\nSuccess Score Statistics:")
print(
    df["success_score"].describe()
)

print("\nSample Success Analysis:")
print(
    df[
        [
            "title",
            "type",
            "rating",
            "listed_in",
            "success_score",
            "success_category"
        ]
    ].head(10)
)

# --------------------------------
# 12. Prepare Data for Machine Learning
# --------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("\n======================================")
print("Preparing Machine Learning Dataset")
print("======================================")

# Select features for the models
model_features = [
    "release_year",
    "content_age",
    "added_year",
    "added_month",
    "genre_count",
    "country_count",
    "duration_value",
    "has_director",
    "is_movie"
]

X = df[model_features].copy()

# Target
y = df["success_category"].copy()

# Convert target categories into numbers
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nFeatures Used:")
print(model_features)

print("\nTarget Classes:")
print(label_encoder.classes_)

print("\nFeature Dataset Shape:")
print(X.shape)

# --------------------------------
# 13. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Records:")
print(X_train.shape[0])

print("Testing Records:")
print(X_test.shape[0])

# --------------------------------
# 14. Train Logistic Regression
# --------------------------------

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Accuracy:")
print(
    round(logistic_accuracy * 100, 2),
    "%"
)

# --------------------------------
# 15. Train Random Forest
# --------------------------------

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\nRandom Forest Accuracy:")
print(
    round(random_forest_accuracy * 100, 2),
    "%"
)

# --------------------------------
# 16. Train Gradient Boosting
# --------------------------------

from sklearn.ensemble import GradientBoostingClassifier

gradient_model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

gradient_model.fit(
    X_train,
    y_train
)

gradient_predictions = gradient_model.predict(
    X_test
)

gradient_accuracy = accuracy_score(
    y_test,
    gradient_predictions
)

print("\nGradient Boosting Accuracy:")
print(
    round(gradient_accuracy * 100, 2),
    "%"
)
# --------------------------------
# 17. Improve Logistic Regression
# --------------------------------

print("\n======================================")
print("Improving Logistic Regression")
print("======================================")

logistic_model_improved = LogisticRegression(
    max_iter=3000,
    solver="saga",
    random_state=42
)

logistic_model_improved.fit(
    X_train,
    y_train
)

improved_predictions = (
    logistic_model_improved.predict(X_test)
)

improved_logistic_accuracy = accuracy_score(
    y_test,
    improved_predictions
)

print("\nImproved Logistic Regression Accuracy:")
print(
    round(improved_logistic_accuracy * 100, 2),
    "%"
)

# --------------------------------
# 18. Compare All Models
# --------------------------------

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        improved_logistic_accuracy,
        random_forest_accuracy,
        gradient_accuracy
    ]
})

model_results["Accuracy (%)"] = (
    model_results["Accuracy"] * 100
).round(2)

print("\n======================================")
print("Model Performance Comparison")
print("======================================")

print(
    model_results[
        ["Model", "Accuracy (%)"]
    ]
)

# --------------------------------
# 19. Select Best Model
# --------------------------------

best_model_row = model_results.loc[
    model_results["Accuracy"].idxmax()
]

print("\n======================================")
print("Best Performing Model")
print("======================================")

print(
    "Model:",
    best_model_row["Model"]
)

print(
    "Accuracy:",
    round(
        best_model_row["Accuracy"] * 100,
        2
    ),
    "%"
)

# --------------------------------
# 20. Generate Automated Business Insights
# --------------------------------

print("\n======================================")
print("Automated Business Insights")
print("======================================")

# --------------------------------
# Insight 1 - Content Type
# --------------------------------

movie_count = (
    df["type"] == "Movie"
).sum()

tv_show_count = (
    df["type"] == "TV Show"
).sum()

if movie_count > tv_show_count:
    dominant_type = "Movies"
    dominant_type_count = movie_count
else:
    dominant_type = "TV Shows"
    dominant_type_count = tv_show_count

print("\nInsight 1:")
print(
    f"{dominant_type} are the dominant content type "
    f"with {dominant_type_count} titles."
)

# --------------------------------
# Insight 2 - Peak Content Year
# --------------------------------

yearly_content = (
    df.groupby("added_year")
    .size()
)

peak_year = (
    yearly_content.idxmax()
)

peak_year_count = (
    yearly_content.max()
)

print("\nInsight 2:")
print(
    f"The highest number of titles was added in "
    f"{int(peak_year)}, with {peak_year_count} titles."
)

# --------------------------------
# Insight 3 - Average Release Year
# --------------------------------

average_release_year = (
    df["release_year"].mean()
)

print("\nInsight 3:")
print(
    f"The average content release year is "
    f"{average_release_year:.1f}."
)

# --------------------------------
# Insight 4 - Success Categories
# --------------------------------

success_distribution = (
    df["success_category"]
    .value_counts()
)

highest_category = (
    success_distribution.idxmax()
)

print("\nInsight 4:")
print(
    f"The largest success category is "
    f"{highest_category}, containing "
    f"{success_distribution.max()} titles."
)

# --------------------------------
# Insight 5 - Best ML Model
# --------------------------------

print("\nInsight 5:")
print(
    f"The best-performing ML model is "
    f"{best_model_row['Model']} "
    f"with an accuracy of "
    f"{best_model_row['Accuracy'] * 100:.2f}%."
)

# --------------------------------
# Insight 6 - Director Availability
# --------------------------------

director_percentage = (
    df["has_director"].mean() * 100
)

print("\nInsight 6:")
print(
    f"{director_percentage:.2f}% of titles "
    f"have director information available."
)

# --------------------------------
# Insight 7 - Content Age
# --------------------------------

average_content_age = (
    df["content_age"].mean()
)

print("\nInsight 7:")
print(
    f"The average content age is "
    f"{average_content_age:.1f} years."
)

# --------------------------------
# 21. Visual Business Report
# --------------------------------

import matplotlib.pyplot as plt

print("\n======================================")
print("Creating Visual Business Report")
print("======================================")

# --------------------------------
# Chart 1 - Model Performance
# --------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    model_results["Model"],
    model_results["Accuracy (%)"]
)

plt.title(
    "Netflix Success Prediction - Model Performance"
)

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy (%)")

plt.xticks(
    rotation=15
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.show()

# --------------------------------
# Chart 2 - Success Category
# --------------------------------

success_counts = (
    df["success_category"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"]
    )
)

plt.figure(figsize=(8, 6))

plt.bar(
    success_counts.index,
    success_counts.values
)

plt.title(
    "Netflix Content Success Categories"
)

plt.xlabel(
    "Success Category"
)

plt.ylabel(
    "Number of Titles"
)

plt.tight_layout()

plt.show()

# --------------------------------
# Chart 3 - Content Added by Year
# --------------------------------

yearly_content = (
    df.groupby("added_year")
    .size()
)

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_content.index,
    yearly_content.values,
    marker="o"
)

plt.title(
    "Netflix Content Added by Year"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Number of Titles Added"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()


print("Visual Business Report Created Successfully")
