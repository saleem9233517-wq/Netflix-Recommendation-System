import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Netflix ML Dashboard",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/netflix_titles.csv")


df = load_data()


# =========================================================
# HEADER
# =========================================================

st.title("🎬 Netflix ML Dashboard")

st.markdown(
    "### Machine Learning Analysis & Insights"
)

st.markdown(
    "Explore Netflix content using recommendation, "
    "classification, clustering, forecasting and analytics."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 Netflix ML Tasks")

selected_task = st.sidebar.selectbox(
    "Select a Task",
    [
        "Dashboard",
        "Content Recommendation",
        "Content Type Prediction",
        "Rating Classification",
        "Content Segmentation",
        "Trend Forecasting",
        "Content Success Analytics"
    ]
)


# =========================================================
# 1. DASHBOARD
# =========================================================

if selected_task == "Dashboard":

    st.subheader("📊 Netflix Dataset Overview")

    total_titles = len(df)

    movies = (
        df["type"]
        .eq("Movie")
        .sum()
    )

    tv_shows = (
        df["type"]
        .eq("TV Show")
        .sum()
    )

    min_year = df["release_year"].min()
    max_year = df["release_year"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Titles",
        f"{total_titles:,}"
    )

    col2.metric(
        "Movies",
        f"{movies:,}"
    )

    col3.metric(
        "TV Shows",
        f"{tv_shows:,}"
    )

    col4.metric(
        "Release Years",
        f"{min_year} - {max_year}"
    )

    st.divider()

    # Content Type Chart

    st.subheader("🎞️ Content Type Distribution")

    type_counts = df["type"].value_counts()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    type_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Number of Titles")
    ax.set_title(
        "Netflix Movies vs TV Shows"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # Dataset Preview

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df[
            [
                "show_id",
                "type",
                "title",
                "country",
                "release_year",
                "rating",
                "duration"
            ]
        ].head(10),
        width="stretch"
    )


# =========================================================
# 2. CONTENT RECOMMENDATION
# =========================================================

elif selected_task == "Content Recommendation":

    st.subheader(
        "🎯 Netflix Content Recommendation"
    )

    st.write(
        "Select a Netflix title and get "
        "5 similar content recommendations."
    )

    # Prepare features

    recommendation_df = df.copy()

    feature_columns = [
        "type",
        "director",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]

    for column in feature_columns:

        recommendation_df[column] = (
            recommendation_df[column]
            .fillna("Unknown")
            .astype(str)
        )

    recommendation_df["content"] = (
        recommendation_df[
            feature_columns
        ]
        .agg(" ".join, axis=1)
        .str.lower()
    )

    # TF-IDF

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        recommendation_df["content"]
    )

    # Title selection

    title_list = sorted(
        recommendation_df["title"]
        .dropna()
        .unique()
    )

    selected_title = st.selectbox(
        "Select a Netflix title",
        title_list
    )

    recommend_button = st.button(
        "🎬 Get Recommendations"
    )

    if recommend_button:

        selected_index = recommendation_df[
            recommendation_df["title"] == selected_title
        ].index[0]

        similarity_scores = cosine_similarity(
            tfidf_matrix[selected_index],
            tfidf_matrix
        ).flatten()

        similar_indices = (
            similarity_scores.argsort()[::-1]
        )

        recommendations = []

        for index in similar_indices:

            if index == selected_index:
                continue

            recommendations.append(
                {
                    "Title": recommendation_df.loc[
                        index,
                        "title"
                    ],
                    "Type": recommendation_df.loc[
                        index,
                        "type"
                    ],
                    "Genre": recommendation_df.loc[
                        index,
                        "listed_in"
                    ],
                    "Similarity": round(
                        similarity_scores[index],
                        2
                    )
                }
            )

            if len(recommendations) == 5:
                break

        st.success(
            f"Recommendations for: {selected_title}"
        )

        result_df = pd.DataFrame(
            recommendations
        )

        st.dataframe(
            result_df,
            width="stretch",
            hide_index=True
        )


# =========================================================
# 3. CONTENT TYPE PREDICTION
# =========================================================

elif selected_task == "Content Type Prediction":

    st.subheader(
        "🤖 Content Type Prediction"
    )

    st.write(
        "Predict whether Netflix content is "
        "a Movie or TV Show."
    )

    st.info(
        "Models: Logistic Regression and Random Forest"
    )

    # Prepare data

    prediction_df = df[
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

    categorical_columns = [
        "director",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]

    for column in categorical_columns:

        prediction_df[column] = (
            prediction_df[column]
            .fillna("Unknown")
            .astype(str)
        )

    prediction_df["type"] = (
        prediction_df["type"]
        .map(
            {
                "Movie": 0,
                "TV Show": 1
            }
        )
    )

    X = prediction_df.drop(
        columns=["type"]
    )

    y = prediction_df["type"]

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    # Train test split

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_encoded,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Models

    logistic_model = LogisticRegression(
        max_iter=1000
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    # Predictions

    logistic_predictions = (
        logistic_model.predict(X_test)
    )

    random_forest_predictions = (
        random_forest_model.predict(X_test)
    )

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )

    random_forest_accuracy = accuracy_score(
        y_test,
        random_forest_predictions
    )

    # Results

    st.subheader(
        "📊 Model Performance"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Logistic Regression",
        f"{logistic_accuracy * 100:.2f}%"
    )

    col2.metric(
        "Random Forest",
        f"{random_forest_accuracy * 100:.2f}%"
    )

    st.divider()

    # Prediction

    st.subheader(
        "🔮 Predict Content Type"
    )

    selected_prediction_title = st.selectbox(
        "Select Title",
        sorted(
            df["title"]
            .dropna()
            .unique()
        ),
        key="prediction_title"
    )

    predict_button = st.button(
        "🔍 Predict Content Type"
    )

    if predict_button:

        original_row = df[
            df["title"] == selected_prediction_title
        ].iloc[0]

        input_row = pd.DataFrame(
            [
                {
                    "director": str(
                        original_row["director"]
                        if pd.notna(
                            original_row["director"]
                        )
                        else "Unknown"
                    ),
                    "country": str(
                        original_row["country"]
                        if pd.notna(
                            original_row["country"]
                        )
                        else "Unknown"
                    ),
                    "release_year":
                        original_row[
                            "release_year"
                        ],
                    "rating": str(
                        original_row["rating"]
                        if pd.notna(
                            original_row["rating"]
                        )
                        else "Unknown"
                    ),
                    "duration": str(
                        original_row["duration"]
                        if pd.notna(
                            original_row["duration"]
                        )
                        else "Unknown"
                    ),
                    "listed_in": str(
                        original_row["listed_in"]
                        if pd.notna(
                            original_row["listed_in"]
                        )
                        else "Unknown"
                    )
                }
            ]
        )

        input_encoded = pd.get_dummies(
            input_row
        )

        input_encoded = input_encoded.reindex(
            columns=X_encoded.columns,
            fill_value=0
        )

        logistic_result = (
            logistic_model.predict(
                input_encoded
            )[0]
        )

        random_forest_result = (
            random_forest_model.predict(
                input_encoded
            )[0]
        )

        logistic_label = (
            "TV Show"
            if logistic_result == 1
            else "Movie"
        )

        random_forest_label = (
            "TV Show"
            if random_forest_result == 1
            else "Movie"
        )

        actual_type = original_row["type"]

        st.success(
            f"Selected Title: "
            f"**{selected_prediction_title}**"
        )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        result_col1.metric(
            "Actual Type",
            actual_type
        )

        result_col2.metric(
            "Logistic Regression",
            logistic_label
        )

        result_col3.metric(
            "Random Forest",
            random_forest_label
        )


# =========================================================
# 4. RATING CLASSIFICATION
# =========================================================

elif selected_task == "Rating Classification":

    st.subheader(
        "⭐ Rating Classification"
    )

    st.write(
        "Predict the Netflix content rating "
        "using Machine Learning."
    )

    st.info(
        "Model: Random Forest"
    )

    # Prepare data

    rating_df = df[
        [
            "type",
            "country",
            "release_year",
            "duration",
            "listed_in",
            "rating"
        ]
    ].copy()

    rating_df["rating"] = (
        rating_df["rating"]
        .fillna("Unknown")
        .astype(str)
    )

    categorical_columns = [
        "type",
        "country",
        "duration",
        "listed_in"
    ]

    for column in categorical_columns:

        rating_df[column] = (
            rating_df[column]
            .fillna("Unknown")
            .astype(str)
        )

    X = rating_df[
        [
            "type",
            "country",
            "release_year",
            "duration",
            "listed_in"
        ]
    ]

    y = rating_df["rating"]

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    # Label encoding

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # Split

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_encoded,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )
    )

    # Random Forest

    rating_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rating_model.fit(
        X_train,
        y_train
    )

    predictions = rating_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Accuracy

    st.subheader(
        "📊 Model Performance"
    )

    st.metric(
        "Random Forest Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.divider()

    # Rating Distribution

    st.subheader(
        "⭐ Netflix Rating Distribution"
    )

    rating_counts = (
        df["rating"]
        .fillna("Unknown")
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    rating_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Titles")
    ax.set_title(
        "Netflix Content Rating Distribution"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # Prediction

    st.subheader(
        "🔮 Predict Content Rating"
    )

    selected_rating_title = st.selectbox(
        "Select a Netflix title",
        sorted(
            df["title"]
            .dropna()
            .unique()
        ),
        key="rating_title"
    )

    predict_rating_button = st.button(
        "⭐ Predict Rating"
    )

    if predict_rating_button:

        original_row = df[
            df["title"] == selected_rating_title
        ].iloc[0]

        input_row = pd.DataFrame(
            [
                {
                    "type": str(
                        original_row["type"]
                    ),
                    "country": str(
                        original_row["country"]
                        if pd.notna(
                            original_row["country"]
                        )
                        else "Unknown"
                    ),
                    "release_year":
                        original_row[
                            "release_year"
                        ],
                    "duration": str(
                        original_row["duration"]
                        if pd.notna(
                            original_row["duration"]
                        )
                        else "Unknown"
                    ),
                    "listed_in": str(
                        original_row["listed_in"]
                        if pd.notna(
                            original_row["listed_in"]
                        )
                        else "Unknown"
                    )
                }
            ]
        )

        input_encoded = pd.get_dummies(
            input_row
        )

        input_encoded = input_encoded.reindex(
            columns=X_encoded.columns,
            fill_value=0
        )

        prediction = rating_model.predict(
            input_encoded
        )[0]

        predicted_rating = (
            label_encoder.inverse_transform(
                [prediction]
            )[0]
        )

        actual_rating = (
            original_row["rating"]
            if pd.notna(
                original_row["rating"]
            )
            else "Unknown"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Actual Rating",
            actual_rating
        )

        col2.metric(
            "Predicted Rating",
            predicted_rating
        )


# =========================================================
# 5. CONTENT SEGMENTATION
# =========================================================

elif selected_task == "Content Segmentation":

    st.subheader(
        "🔵 Netflix Content Segmentation"
    )

    st.write(
        "Group Netflix content into clusters using "
        "K-Means clustering."
    )

    # Prepare data

    segmentation_df = df[
        [
            "type",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in"
        ]
    ].copy()

    categorical_columns = [
        "type",
        "country",
        "rating",
        "duration",
        "listed_in"
    ]

    for column in categorical_columns:

        segmentation_df[column] = (
            segmentation_df[column]
            .fillna("Unknown")
            .astype(str)
        )

    df_encoded = pd.get_dummies(
        segmentation_df,
        columns=categorical_columns,
        drop_first=True
    )

    df_encoded = df_encoded.astype(float)

    # K-Means

    number_of_clusters = 5

    kmeans_model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = (
        kmeans_model.fit_predict(
            df_encoded
        )
    )

    segmentation_df["cluster"] = (
        cluster_labels
    )

    # Cluster distribution

    st.subheader(
        "📊 Cluster Distribution"
    )

    cluster_counts = (
        segmentation_df["cluster"]
        .value_counts()
        .sort_index()
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    cluster_columns = [
        col1,
        col2,
        col3,
        col4,
        col5
    ]

    for i, column in enumerate(
        cluster_columns
    ):

        column.metric(
            f"Cluster {i}",
            f"{cluster_counts.get(i, 0):,}"
        )

    st.divider()

    # PCA

    st.subheader(
        "🔬 Cluster Visualization"
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_features = pca.fit_transform(
        df_encoded
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    scatter = ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        c=segmentation_df["cluster"],
        alpha=0.6
    )

    ax.set_title(
        "Netflix Content Segmentation"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    plt.tight_layout()

    st.pyplot(fig)

    explained_variance = (
        pca.explained_variance_ratio_.sum()
        * 100
    )

    st.info(
        f"PCA explained variance: "
        f"{explained_variance:.2f}%"
    )

    st.divider()

    # Cluster table

    st.subheader(
        "📋 Cluster Summary"
    )

    cluster_summary = []

    for cluster_id in sorted(
        segmentation_df["cluster"].unique()
    ):

        cluster_data = segmentation_df[
            segmentation_df["cluster"]
            == cluster_id
        ]

        movie_count = (
            cluster_data["type"]
            .eq("Movie")
            .sum()
        )

        tv_count = (
            cluster_data["type"]
            .eq("TV Show")
            .sum()
        )

        average_year = (
            cluster_data["release_year"]
            .mean()
        )

        cluster_summary.append(
            {
                "Cluster": cluster_id,
                "Titles": len(cluster_data),
                "Movies": movie_count,
                "TV Shows": tv_count,
                "Average Release Year":
                    round(
                        average_year,
                        2
                    )
            }
        )

    st.dataframe(
        pd.DataFrame(
            cluster_summary
        ),
        width="stretch",
        hide_index=True
    )


# =========================================================
# 6. TREND FORECASTING
# =========================================================

elif selected_task == "Trend Forecasting":

    st.subheader(
        "📈 Netflix Content Trend Forecasting"
    )

    st.write(
        "Analyze historical Netflix content additions "
        "and forecast future content trends."
    )

    # Prepare dates

    trend_df = df.copy()

    trend_df["date_added"] = pd.to_datetime(
        trend_df["date_added"],
        errors="coerce"
    )

    trend_df["added_year"] = (
        trend_df["date_added"].dt.year
    )

    trend_df["added_month"] = (
        trend_df["date_added"].dt.month
    )

    trend_df["release_year"] = (
        pd.to_numeric(
            trend_df["release_year"],
            errors="coerce"
        )
    )

    # Yearly content

    yearly_content = (
        trend_df
        .dropna(subset=["added_year"])
        .groupby("added_year")
        .size()
        .reset_index(
            name="content_count"
        )
    )

    yearly_content["added_year"] = (
        yearly_content["added_year"]
        .astype(int)
    )

    # Peak year

    peak_row = yearly_content.loc[
        yearly_content[
            "content_count"
        ].idxmax()
    ]

    peak_year = int(
        peak_row["added_year"]
    )

    peak_count = int(
        peak_row["content_count"]
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Peak Year",
        peak_year
    )

    col2.metric(
        "Titles Added",
        f"{peak_count:,}"
    )

    st.divider()

    # Historical Trend

    st.subheader(
        "📊 Netflix Content Added by Year"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        yearly_content["added_year"],
        yearly_content["content_count"],
        marker="o"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Number of Titles"
    )

    ax.set_title(
        "Netflix Content Added by Year"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # Forecast model

    st.subheader(
        "🔮 Future Content Forecast"
    )

    X = yearly_content[
        ["added_year"]
    ]

    y = yearly_content[
        "content_count"
    ]

    forecast_model = LinearRegression()

    forecast_model.fit(
        X,
        y
    )

    future_years = pd.DataFrame(
        {
            "added_year": [
                2022,
                2023,
                2024,
                2025
            ]
        }
    )

    future_predictions = (
        forecast_model.predict(
            future_years
        )
    )

    future_predictions = (
        future_predictions.round()
        .astype(int)
    )

    forecast_results = future_years.copy()

    forecast_results[
        "Forecasted Titles"
    ] = future_predictions

    st.dataframe(
        forecast_results,
        width="stretch",
        hide_index=True
    )

    # Forecast chart

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        yearly_content["added_year"],
        yearly_content["content_count"],
        marker="o",
        label="Historical"
    )

    ax.plot(
        future_years["added_year"],
        future_predictions,
        marker="o",
        linestyle="--",
        label="Forecast"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Number of Titles"
    )

    ax.set_title(
        "Netflix Content Trend Forecast"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Forecast generated using Linear Regression "
        "on yearly content additions."
    )


# =========================================================
# 7. CONTENT SUCCESS ANALYTICS
# =========================================================

elif selected_task == "Content Success Analytics":

    st.subheader(
        "🏆 Netflix Content Success Analytics"
    )

    st.write(
        "Analyze Netflix content using engineered "
        "features and a data-driven success proxy."
    )

    # --------------------------------
    # Prepare Features
    # --------------------------------

    analytics_data = df.copy()

    analytics_data["date_added"] = (
        pd.to_datetime(
            analytics_data["date_added"],
            errors="coerce"
        )
    )

    analytics_data["added_year"] = (
        analytics_data["date_added"]
        .dt.year
    )

    analytics_data["added_month"] = (
        analytics_data["date_added"]
        .dt.month
    )

    reference_year = int(
        analytics_data["release_year"].max()
    )

    analytics_data["content_age"] = (
        reference_year
        - analytics_data["release_year"]
    )

    analytics_data["genre_count"] = (
        analytics_data["listed_in"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .apply(len)
    )

    analytics_data["country_count"] = (
        analytics_data["country"]
        .fillna("Unknown")
        .astype(str)
        .str.split(",")
        .apply(len)
    )

    analytics_data["duration_value"] = (
        analytics_data["duration"]
        .fillna("0")
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False
        )
        .fillna(0)
        .astype(float)
    )

    analytics_data["has_director"] = (
        analytics_data["director"]
        .notna()
        .astype(int)
    )

    analytics_data["is_movie"] = (
        analytics_data["type"]
        == "Movie"
    ).astype(int)

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

    feature_data = analytics_data[
        engineered_features
    ].copy()

    feature_data = feature_data.fillna(0)

    # --------------------------------
    # Success Proxy
    # --------------------------------

    genre_frequency = (
        analytics_data["listed_in"]
        .fillna("Unknown")
        .value_counts()
    )

    analytics_data[
        "genre_popularity"
    ] = (
        analytics_data["listed_in"]
        .fillna("Unknown")
        .map(genre_frequency)
    )

    country_frequency = (
        analytics_data["country"]
        .fillna("Unknown")
        .value_counts()
    )

    analytics_data[
        "country_popularity"
    ] = (
        analytics_data["country"]
        .fillna("Unknown")
        .map(country_frequency)
    )

    rating_frequency = (
        analytics_data["rating"]
        .fillna("Unknown")
        .value_counts()
    )

    analytics_data[
        "rating_popularity"
    ] = (
        analytics_data["rating"]
        .fillna("Unknown")
        .map(rating_frequency)
    )

    analytics_data[
        "success_score"
    ] = (
        analytics_data[
            "genre_popularity"
        ]
        + analytics_data[
            "country_popularity"
        ]
        + analytics_data[
            "rating_popularity"
        ]
    )

    # Three categories

    success_rank = (
        analytics_data["success_score"]
        .rank(method="first")
    )

    analytics_data[
        "success_category"
    ] = pd.qcut(
        success_rank,
        q=3,
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------
    # Success Metrics
    # --------------------------------

    st.subheader(
        "📊 Success Analytics"
    )

    average_success = (
        analytics_data[
            "success_score"
        ].mean()
    )

    average_age = (
        feature_data[
            "content_age"
        ].mean()
    )

    movies_count = (
        analytics_data["type"]
        .eq("Movie")
        .sum()
    )

    best_year = (
        analytics_data[
            "added_year"
        ]
        .value_counts()
        .idxmax()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Average Success Score",
        f"{average_success:.2f}"
    )

    col2.metric(
        "Average Content Age",
        f"{average_age:.1f} years"
    )

    col3.metric(
        "Movies",
        f"{movies_count:,}"
    )

    col4.metric(
        "Peak Added Year",
        int(best_year)
    )

    st.divider()

    # --------------------------------
    # Model Preparation
    # --------------------------------

    success_encoder = LabelEncoder()

    success_y = (
        success_encoder.fit_transform(
            analytics_data[
                "success_category"
            ]
        )
    )

    success_X = feature_data

    X_train, X_test, y_train, y_test = (
        train_test_split(
            success_X,
            success_y,
            test_size=0.20,
            random_state=42,
            stratify=success_y
        )
    )

    # Logistic Regression

    logistic_success = LogisticRegression(
        max_iter=3000,
        solver="saga",
        random_state=42
    )

    logistic_success.fit(
        X_train,
        y_train
    )

    logistic_predictions = (
        logistic_success.predict(
            X_test
        )
    )

    logistic_accuracy = (
        accuracy_score(
            y_test,
            logistic_predictions
        )
    )

    # Random Forest

    random_forest_success = (
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )

    random_forest_success.fit(
        X_train,
        y_train
    )

    rf_predictions = (
        random_forest_success.predict(
            X_test
        )
    )

    rf_accuracy = (
        accuracy_score(
            y_test,
            rf_predictions
        )
    )

    # Gradient Boosting

    gradient_model = (
        GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    )

    gradient_model.fit(
        X_train,
        y_train
    )

    gradient_predictions = (
        gradient_model.predict(
            X_test
        )
    )

    gradient_accuracy = (
        accuracy_score(
            y_test,
            gradient_predictions
        )
    )

    # --------------------------------
    # Model Comparison
    # --------------------------------

    st.subheader(
        "🤖 Success Prediction Models"
    )

    model_results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting"
            ],
            "Accuracy": [
                logistic_accuracy,
                rf_accuracy,
                gradient_accuracy
            ]
        }
    )

    model_results[
        "Accuracy (%)"
    ] = (
        model_results["Accuracy"]
        * 100
    )

    best_model_row = model_results.loc[
        model_results["Accuracy"].idxmax()
    ]

    best_model_name = (
        best_model_row["Model"]
    )

    best_model_accuracy = (
        best_model_row["Accuracy (%)"]
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Model",
        best_model_name
    )

    col2.metric(
        "Best Accuracy",
        f"{best_model_accuracy:.2f}%"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        model_results["Model"],
        model_results["Accuracy (%)"]
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------
    # Success Categories
    # --------------------------------

    st.subheader(
        "🏆 Success Category Distribution"
    )

    category_counts = (
        analytics_data[
            "success_category"
        ]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High"
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    category_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Success Category"
    )

    ax.set_ylabel(
        "Number of Titles"
    )

    ax.set_title(
        "Netflix Success Categories"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------
    # Content Trend
    # --------------------------------

    st.subheader(
        "📈 Netflix Content Added by Year"
    )

    yearly_success = (
        analytics_data
        .dropna(
            subset=["added_year"]
        )
        .groupby("added_year")
        .size()
        .reset_index(
            name="content_count"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        yearly_success["added_year"],
        yearly_success["content_count"],
        marker="o"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Number of Titles"
    )

    ax.set_title(
        "Netflix Content Added by Year"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------
    # Automated Insights
    # --------------------------------

    st.subheader(
        "💡 Automated Insights"
    )

    director_percentage = (
        analytics_data[
            "has_director"
        ].mean()
        * 100
    )

    average_release_year = (
        analytics_data[
            "release_year"
        ].mean()
    )

    st.write(
        f"• Movies are the dominant content type "
        f"with **{movies_count:,} titles**."
    )

    st.write(
        f"• The peak content addition year was "
        f"**{int(best_year)}**."
    )

    st.write(
        f"• Average release year is approximately "
        f"**{average_release_year:.1f}**."
    )

    st.write(
        f"• Approximately **{director_percentage:.0f}%** "
        f"of titles contain director information."
    )

    st.write(
        f"• Average content age is "
        f"**{average_age:.1f} years**."
    )

    st.warning(
        "Success Score is a dataset-based proxy using "
        "genre, country and rating frequency. The dataset "
        "does not contain actual views, revenue, likes or "
        "audience engagement data."
    )
