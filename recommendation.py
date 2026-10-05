import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# 1. Load Netflix Dataset
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")


# --------------------------------
# 2. Select Relevant Columns
# --------------------------------

df = df[
    [
        "show_id",
        "type",
        "title",
        "director",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in"
    ]
].copy()


# --------------------------------
# 3. Handle Missing Values
# --------------------------------

text_columns = [
    "director",
    "country",
    "date_added",
    "rating",
    "duration",
    "listed_in"
]

for column in text_columns:
    df[column] = df[column].fillna("")
    df[column] = df[column].astype(str)


# --------------------------------
# 4. Create Content Feature
# --------------------------------

df["content"] = (
    df["type"] + " " +
    df["director"] + " " +
    df["country"] + " " +
    df["rating"] + " " +
    df["duration"] + " " +
    df["listed_in"]
)

df["content"] = df["content"].str.lower()


# --------------------------------
# 5. Remove Duplicate Titles
# --------------------------------

df = df.drop_duplicates(subset="title")
df = df.reset_index(drop=True)


# --------------------------------
# 6. TF-IDF Vectorization
# --------------------------------

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(df["content"])


# --------------------------------
# 7. Cosine Similarity
# --------------------------------

similarity_matrix = cosine_similarity(tfidf_matrix)


# --------------------------------
# 8. Recommendation Function
# --------------------------------

def recommend(title, number_of_recommendations=5):

    # Find title
    matches = df[df["title"].str.lower() == title.lower()]

    if matches.empty:
        print("\nTitle not found.")
        return

    # Get index
    title_index = matches.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[title_index])
    )

    # Sort by similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top recommendations
    recommendations = []

    for index, score in similarity_scores:
        if index != title_index:
            recommendations.append(
                (index, score)
            )

        if len(recommendations) == number_of_recommendations:
            break

    # Display results
    print("\n======================================")
    print("Netflix Recommendations")
    print("======================================")

    print("\nSelected Title:")
    print(df.iloc[title_index]["title"])

    print("\nRecommended Titles:")

    for position, (index, score) in enumerate(
        recommendations,
        start=1
    ):
        print(
            f"{position}. "
            f"{df.iloc[index]['title']} "
            f"(Similarity: {score:.2f})"
        )


# --------------------------------
# 9. Test Recommendation System
# --------------------------------

recommend("Dick Johnson Is Dead", 5)


# --------------------------------
# 10. Evaluate Recommendation Quality
# --------------------------------

def evaluate_recommendations(title, number_of_recommendations=5):

    matches = df[df["title"].str.lower() == title.lower()]

    if matches.empty:
        print("\nTitle not found.")
        return

    title_index = matches.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[title_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top recommendations
    recommendations = []

    for index, score in similarity_scores:
        if index != title_index:
            recommendations.append((index, score))

        if len(recommendations) == number_of_recommendations:
            break

    # Calculate average similarity
    average_similarity = sum(
        score for index, score in recommendations
    ) / len(recommendations)

    # Genre overlap
    selected_genres = set(
        df.iloc[title_index]["listed_in"].lower().split(", ")
    )

    genre_matches = 0

    for index, score in recommendations:

        recommended_genres = set(
            df.iloc[index]["listed_in"].lower().split(", ")
        )

        if selected_genres.intersection(recommended_genres):
            genre_matches += 1

    genre_overlap_rate = (
        genre_matches / len(recommendations)
    ) * 100

    # Display evaluation
    print("\n======================================")
    print("Recommendation Evaluation")
    print("======================================")

    print("\nSelected Title:")
    print(df.iloc[title_index]["title"])

    print(
        f"\nAverage Similarity Score: "
        f"{average_similarity:.2f}"
    )

    print(
        f"Genre Overlap Rate: "
        f"{genre_overlap_rate:.2f}%"
    )

    print(
        f"Recommendations Evaluated: "
        f"{len(recommendations)}"
    )


# --------------------------------
# 11. Test Evaluation
# --------------------------------

evaluate_recommendations(
    "Dick Johnson Is Dead",
    5
)