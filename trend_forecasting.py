import pandas as pd

# --------------------------------
# Netflix Trend Forecasting Model
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")

print("======================================")
print("Netflix Trend Forecasting Model")
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
# 2. Prepare Release Year Feature
# --------------------------------

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

# --------------------------------
# 3. Remove Records Without Dates
# --------------------------------

df_time = df.dropna(
    subset=["date_added", "added_year"]
).copy()

# Convert year to integer
df_time["added_year"] = (
    df_time["added_year"]
    .astype(int)
)

# --------------------------------
# 4. Display Time-Based Features
# --------------------------------

print("\n======================================")
print("Time-Based Features Prepared")
print("======================================")

print("\nTime-Based Dataset Shape:")
print(df_time.shape)

print("\nAvailable Years:")
print(
    sorted(
        df_time["added_year"].unique()
    )
)

print("\nFirst 5 Time-Based Records:")

print(
    df_time[
        [
            "title",
            "type",
            "date_added",
            "added_year",
            "added_month",
            "release_year"
        ]
    ].head()
)
# --------------------------------
# 5. Analyze Historical Release Patterns
# --------------------------------

yearly_content = (
    df_time
    .groupby("added_year")
    .size()
    .reset_index(name="content_count")
)

print("\n======================================")
print("Historical Release Pattern")
print("======================================")

print("\nContent Added Per Year:")

print(
    yearly_content.to_string(index=False)
)

# --------------------------------
# 6. Movie and TV Show Trends
# --------------------------------

yearly_type = (
    df_time
    .groupby(
        ["added_year", "type"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\n======================================")
print("Movie vs TV Show Trend")
print("======================================")

print(
    yearly_type.to_string()
)

# --------------------------------
# 7. Peak Content Addition Year
# --------------------------------

peak_year = (
    yearly_content.loc[
        yearly_content["content_count"].idxmax()
    ]
)

print("\n======================================")
print("Peak Content Addition")
print("======================================")

print(
    "\nPeak Year:",
    int(peak_year["added_year"])
)

print(
    "Titles Added:",
    int(peak_year["content_count"])
)

# --------------------------------
# 8. Build Forecasting Model
# --------------------------------

from sklearn.linear_model import LinearRegression

# Prepare training data
X = yearly_content[["added_year"]]
y = yearly_content["content_count"]

# Create Linear Regression model
forecast_model = LinearRegression()

forecast_model.fit(
    X,
    y
)

print("\n======================================")
print("Forecasting Model Trained")
print("======================================")

print("\nModel Coefficient:")
print(
    round(
        forecast_model.coef_[0],
        2
    )
)

print("\nModel Intercept:")
print(
    round(
        forecast_model.intercept_,
        2
    )
)

print("\nTraining R² Score:")
print(
    round(
        forecast_model.score(X, y),
        4
    )
)

# --------------------------------
# 9. Generate Future Predictions
# --------------------------------

future_years = pd.DataFrame({
    "added_year": [2022, 2023, 2024, 2025]
})

future_predictions = forecast_model.predict(
    future_years[["added_year"]]
)

# Negative predictions ko zero se neeche nahi jane dena
future_predictions = future_predictions.clip(min=0)

future_years["predicted_content_count"] = (
    future_predictions.round().astype(int)
)

print("\n======================================")
print("Future Content Predictions")
print("======================================")

print(
    future_years.to_string(index=False)
)

# --------------------------------
# 10. Evaluate Forecasting Accuracy
# --------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predictions for historical data
historical_predictions = forecast_model.predict(
    X
)

# Calculate evaluation metrics
mae = mean_absolute_error(
    y,
    historical_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        historical_predictions
    )
)

r2 = forecast_model.score(
    X,
    y
)

print("\n======================================")
print("Forecasting Model Evaluation")
print("======================================")

print(
    "\nMean Absolute Error (MAE):",
    round(mae, 2)
)

print(
    "Root Mean Squared Error (RMSE):",
    round(rmse, 2)
)

print(
    "R² Score:",
    round(r2, 4)
)

