# Netflix Recommendation System

A Machine Learning project developed during my **4-Week Machine Learning Internship at Auspify Technologies**.

This project analyzes the Netflix titles dataset and implements multiple Machine Learning techniques for recommendation, classification, clustering, trend forecasting, and content success analytics.

---

## Project Overview

The project contains six Machine Learning tasks:

1. **Netflix Content Recommendation System**
2. **Content Type Prediction**
3. **Rating Classification**
4. **Content Segmentation**
5. **Trend Forecasting**
6. **Content Success Analytics**

The project uses the Netflix titles dataset containing **8,790 records and 10 features**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- TF-IDF Vectorization
- Cosine Similarity
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Means Clustering
- PCA
- Linear Regression

---

## Project Structure

```text
Netflix-Recommendation-System/
│
├── data/
│   └── netflix_titles.csv
│
├── screenshots/
│   ├── task1_recommendations.png
│   ├── task1_evaluation.png
│   ├── task2_model_results.png
│   ├── task3_rating_distribution.png
│   ├── task3_model_results.png
│   ├── task4_cluster_distribution.png
│   ├── task4_clusters.png
│   ├── task5_yearly_trend.png
│   ├── task5_forecast.png
│   ├── task6_model_comparison.png
│   ├── task6_success_categories.png
│   └── task6_content_trend.png
│
├── recommendation.py
├── content_type_prediction.py
├── rating_classification.py
├── content_segmentation.py
├── trend_forecasting.py
├── content_success_analytics.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Task 1 — Netflix Content Recommendation System

A content-based recommendation system was developed using:

- TF-IDF Vectorization
- Cosine Similarity

Content information such as type, director, country, rating, duration, and genres was combined to create content features.

### Results

**TF-IDF Matrix:**

```text
8787 × 6769
```

Example recommendations for **Dick Johnson Is Dead**:

| Rank | Recommended Title | Similarity |
|---|---|---:|
| 1 | Juanita | 0.42 |
| 2 | Gender Revolution: A Journey with Katie Couric | 0.40 |
| 3 | S.W.A.T. | 0.39 |
| 4 | Roll With Me | 0.37 |
| 5 | Ghost Rider | 0.34 |

### Evaluation

- Average Similarity Score: **0.38**
- Genre Overlap Rate: **40%**
- Recommendations Evaluated: **5**

---

# Task 2 — Content Type Prediction

This task predicts whether a Netflix title is a:

- Movie
- TV Show

Machine Learning models used:

- Logistic Regression
- Random Forest

### Dataset Distribution

```text
Movies   : 6126
TV Shows : 2664
```

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | **98.63%** |
| Random Forest | **98.58%** |

Logistic Regression achieved the highest accuracy for this task.

---

# Task 3 — Rating Classification

This task predicts Netflix content rating categories using Machine Learning.

The dataset contains **14 different rating categories**, including:

- TV-MA
- TV-14
- TV-PG
- R
- PG-13
- TV-Y7
- TV-Y
- PG
- TV-G
- NR
- G
- TV-Y7-FV
- NC-17
- UR

### Models Used

- Logistic Regression
- Random Forest

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | **41.07%** |
| Random Forest | **50.17%** |

Random Forest performed better than Logistic Regression for rating classification.

The lower accuracy is influenced by the large number of rating classes and strong class imbalance, especially for rare categories.

---

# Task 4 — Netflix Content Segmentation

K-Means clustering was used to group Netflix content into different segments based on content characteristics.

### Methodology

- Feature encoding using One-Hot Encoding
- K-Means Clustering
- 5 clusters
- PCA for 2D visualization

### Cluster Distribution

| Cluster | Content Count |
|---|---:|
| Cluster 0 | 2181 |
| Cluster 1 | 238 |
| Cluster 2 | 5652 |
| Cluster 3 | 667 |
| Cluster 4 | 52 |

The clusters represent different groups of Netflix content based on characteristics such as content type, country, rating, duration, genre, and release year.

---

# Task 5 — Trend Forecasting

This task analyzes Netflix content addition trends over time and forecasts future content additions.

### Peak Year

**2019**

Number of titles added:

```text
2016
```

### Forecast Model

Linear Regression was used for forecasting.

Training R²:

```text
0.7528
```

### Forecast

| Year | Predicted Content |
|---|---:|
| 2022 | 1895 |
| 2023 | 2064 |
| 2024 | 2233 |
| 2025 | 2402 |

The model indicates an increasing trend in predicted content additions.

---

# Task 6 — Content Success Analytics

This task develops a Machine Learning-based analytics system for estimating content success using engineered features.

### Engineered Features

- Release Year
- Content Age
- Added Year
- Added Month
- Genre Count
- Country Count
- Duration
- Director Availability
- Content Type

### Success Proxy

Because the Netflix titles dataset does not contain direct engagement metrics such as:

- Views
- Revenue
- Likes
- Watch time
- User ratings

a **content success proxy**