"""


Available Functions
-------------------
[Public]


[Private]

-------------------
"""


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def extract_features(data_dict):
    """
    Extracts relevant features from the session dictionary into a DataFrame.
    """
    sessions = []

    for session_id, session_data in data_dict.items():
        walking = session_data["Walking"]
        sitting = session_data["Sitting"]
        standing = session_data["Standing"]
        proportions = session_data["Proportions"]

        session_features = {
            "id": session_id,
            "total_distance": walking["total_distance"],
            "avg_speed": walking["avg_speed"],
            "n_steps": walking["n_steps"],
            "walking_prop": proportions["walking_proportion"],
            "n_segments_walking": walking["n_segments"],
            "sitting_rotation": sitting["rotation_percent"],
            "sitting_prop": proportions["sitting_proportion"],
            "n_segments_sitting": walking["n_segments"],
            "standing_rotation": standing["rotation_percent"],
            "standing_prop": proportions["standing_proportion"],
            "n_segments_standing": standing["n_segments"]
        }

        sessions.append(session_features)

    return pd.DataFrame(sessions)


def normalize_features(df, drop_cols=["id"]):
    """
    Normalizes the feature columns using StandardScaler.
    Returns the scaled values and the column names.
    """
    features = df.drop(columns=drop_cols)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, features.columns


def run_kmeans_elbow(X_scaled, k_range=range(1, 20), plot=True):
    """
    Applies the elbow method to choose the optimal number of clusters
    and runs K-Means with that value of k.

    Parameters:ar
        X_scaled: array-like, the normalized feature data
        k_range: iterable of integers, range of k values to test
        plot: bool, whether to plot the elbow curve

    Returns:
        best_kmeans: KMeans object fitted with optimal k
        best_labels: cluster labels
        optimal_k: the chosen number of clusters
    """
    n_samples = X_scaled.shape[0]
    valid_k_range = range(1, n_samples + 1)

    if n_samples < 2:
        raise ValueError("Pelo menos 2 amostras são necessárias para clustering.")

    inertias = []

    for k in valid_k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(list(valid_k_range), inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

    if len(inertias) == 1:
        optimal_k = 1
    else:
        deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        optimal_k = deltas.index(max(deltas)) + 1  # +1 because k starts at 1

    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    best_labels = best_kmeans.fit_predict(X_scaled)

    return best_kmeans, best_labels, optimal_k




def assign_clusters(df, labels):
    """
    Adds cluster labels to the DataFrame.
    """
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = labels
    return df_with_clusters


def summarize_clusters(df_with_clusters):
    """
    Prints the mean of each feature per cluster.
    """
    return df_with_clusters.groupby("cluster").mean(numeric_only=True)


# -----------------------
# Example usage in main()
# -----------------------
# Load JSON data
import json
with open(r"D:\Mariana\1º ano Mestrado - 2º semestre\Prevoccupai\data\metrics\all_metrics.json", "r") as f:
    data = json.load(f)


df = extract_features(data)
X_scaled, _ = normalize_features(df)
best_kmeans, labels, optimal_k = run_kmeans_elbow(X_scaled)
df_clustered = assign_clusters(df, labels)
"""
print("Cluster assignment per session:")
print(df_clustered[["id", "cluster"]])

print("\nCluster-wise average features:")
print(summarize_clusters(df_clustered))
"""

# Example:
# if __name__ == "__main__":
#     from my_data_file import data_dict
#     main(data_dict)
