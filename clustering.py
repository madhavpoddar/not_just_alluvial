from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

from tqdm import tqdm

import random


def add_clustering_data(df, n_dim_list, dim_reduced_data, clustering_alg="kmeans"):
    col_names = []
    print("Performing Clustering")

    # n_clusters_list = [2,3,4,5,6,7,8]
    # n_dim = 2

    n_clusters = 25

    if clustering_alg == "kmeans":
        for i, n_dim in tqdm(enumerate(n_dim_list)):
            # for i, n_clusters in tqdm(enumerate(n_clusters_list)):

            # col_name = str(n_clusters)
            col_name = str(n_dim)
            col_names.append(col_name)

            # For testing purpose
            # n_clusters = random.choice([10, 3, 5, 7])
            df[col_name] = cl_kmeans(dim_reduced_data, n_dim, n_clusters)
            # For testing purpose
            # df[col_name] += 100
    print("Clustering Done")
    return col_names


def cl_kmeans(projected, n_dim=2, n_clusters=10):
    dim_red_array = projected[:, :n_dim]
    dim_red_array = StandardScaler().fit_transform(dim_red_array)
    kmeans = KMeans(
        init="random", n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42
    )
    kmeans.fit(dim_red_array)
    return kmeans.labels_


def cl_dbscan(projected, n_dim=2, n_clusters=10):
    dim_red_array = projected[:, :n_dim]
    db = DBSCAN(eps=0.999, min_samples=10)
    db.fit(dim_red_array)
    return db.labels_
