import pandas as pd

# --------------------------------
# Netflix Content Segmentation
# --------------------------------

df = pd.read_csv("data/netflix_titles.csv")

print("======================================")
print("Netflix Content Segmentation")
print("======================================")

print("\nOriginal Dataset Shape:")
print(df.shape)

# --------------------------------
# 1. Select Relevant Features
# --------------------------------

df = df[
    [
        "type",
        "country",
        "release_year",
        "rating",
        "duration",
        "listed_in"
    ]
].copy()

# --------------------------------
# 2. Handle Missing Values
# --------------------------------

categorical_columns = [
    "type",
    "country",
    "rating",
    "duration",
    "listed_in"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")
    df[column] = df[column].astype(str)

# --------------------------------
# 3. Encode Categorical Features
# --------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# --------------------------------
# 4. Convert Features to Numeric
# --------------------------------

df_encoded = df_encoded.astype(float)

# --------------------------------
# 5. Display Prepared Features
# --------------------------------

print("\n======================================")
print("Features Prepared")
print("======================================")

print("\nFeature Shape:")
print(df_encoded.shape)

print("\nSelected Features:")
print(df.columns.tolist())

print("\nFirst 5 Encoded Records:")
print(df_encoded.head())



# --------------------------------
# 6. Apply K-Means Clustering
# --------------------------------

from sklearn.cluster import KMeans

# Number of clusters
number_of_clusters = 5

kmeans_model = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_model.fit_predict(
    df_encoded
)

# Add cluster labels to original prepared data
df["cluster"] = cluster_labels

# --------------------------------
# 7. Display Clustering Results
# --------------------------------

print("\n======================================")
print("K-Means Clustering Completed")
print("======================================")

print("\nNumber of Clusters:", number_of_clusters)

print("\nCluster Distribution:")
print(df["cluster"].value_counts().sort_index())

# --------------------------------
# 8. Identify Content Groups
# --------------------------------

print("\n======================================")
print("Cluster Content Groups")
print("======================================")

for cluster_number in sorted(df["cluster"].unique()):

    cluster_data = df[
        df["cluster"] == cluster_number
    ]

    print("\n--------------------------------------")
    print("Cluster:", cluster_number)
    print("--------------------------------------")

    print(
        "Number of Titles:",
        len(cluster_data)
    )

    print(
        "Average Release Year:",
        round(
            cluster_data["release_year"].mean(),
            2
        )
    )

    print("\nMost Common Content Type:")
    print(
        cluster_data["type"]
        .value_counts()
        .head(1)
    )

    print("\nMost Common Rating:")
    print(
        cluster_data["rating"]
        .value_counts()
        .head(1)
    )

    print("\nTop Genres:")
    print(
        cluster_data["listed_in"]
        .value_counts()
        .head(5)
    )

# --------------------------------
# 9. Visualize Clusters using PCA
# --------------------------------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce features to 2 dimensions
pca = PCA(
    n_components=2,
    random_state=42
)

pca_features = pca.fit_transform(df_encoded)

# Create cluster visualization
plt.figure(figsize=(10, 7))

plt.scatter(
    pca_features[:, 0],
    pca_features[:, 1],
    c=df["cluster"],
    alpha=0.6
)

plt.title("Netflix Content Segmentation")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")

plt.tight_layout()
plt.show()

print("\n======================================")
print("Cluster Visualization Completed")
print("======================================")

print(
    "\nExplained Variance Ratio:",
    pca.explained_variance_ratio_
)

print(
    "\nTotal Explained Variance:",
    round(
        pca.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

# --------------------------------
# 10. Interpret Cluster Characteristics
# --------------------------------

print("\n======================================")
print("Cluster Interpretation")
print("======================================")

for cluster_number in sorted(df["cluster"].unique()):

    cluster_data = df[
        df["cluster"] == cluster_number
    ]

    common_type = (
        cluster_data["type"]
        .value_counts()
        .idxmax()
    )

    common_rating = (
        cluster_data["rating"]
        .value_counts()
        .idxmax()
    )

    common_genre = (
        cluster_data["listed_in"]
        .value_counts()
        .idxmax()
    )

    average_year = cluster_data["release_year"].mean()

    print("\n--------------------------------------")
    print("Cluster:", cluster_number)
    print("--------------------------------------")

    print(
        "Content Count:",
        len(cluster_data)
    )

    print(
        "Dominant Content Type:",
        common_type
    )

    print(
        "Dominant Rating:",
        common_rating
    )

    print(
        "Most Common Genre:",
        common_genre
    )

    print(
        "Average Release Year:",
        round(average_year, 2)
    )

print("\n======================================")
print("Netflix Content Segmentation Completed")
print("======================================")