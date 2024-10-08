import numpy as np
import os
import pandas as pd
import warnings
from tqdm import tqdm

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from nja_helper_functions import read_csv_file, df_remove_col


def calc_DR_data(df):
    print("Performing Dimensionality Reduction")
    pca = PCA(n_components=df.shape[1]).fit(df)
    # visualize_pca_explained_variance_ratio_(pca)
    dim_reduced_data = pca.fit_transform(df)
    scaler = StandardScaler()
    dim_reduced_data = scaler.fit_transform(dim_reduced_data)
    # visualize_pca_projected_2d(projected)
    print("Dimensionality Reduction Done")
    return dim_reduced_data


def add_clustering_data(df, n_dim_list, dim_reduced_data, clustering_alg="kmeans"):
    col_names = []
    print("Performing Clustering")
    n_clusters = 25
    if clustering_alg == "kmeans":
        for i, n_dim in tqdm(enumerate(n_dim_list)):
            col_name = str(n_dim)
            col_names.append(col_name)
            df[col_name] = cl_kmeans(dim_reduced_data, n_dim, n_clusters)
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


def get_sample_df(which_data: str = "MNIST_DiffDim_KMeans25"):
    if which_data == "MNIST_DiffDim_KMeans25":
        # Reading data and initial modifications
        df = read_csv_file("mnist_test.csv")
        label_col_name = "label"
        df_data_space = df_remove_col(df, label_col_name)

        # perform DR and clustering
        dim_reduced_data = calc_DR_data(df_data_space)
        dim_reduced_data_pca_df = pd.DataFrame(
            dim_reduced_data,
            columns=["pca_" + str(i) for i in range(dim_reduced_data.shape[1])],
        )
        df = pd.concat([df, dim_reduced_data_pca_df], axis=1)

        #  perform TSNE
        # tsne784_2 = TSNE(n_components=2, random_state=0).fit_transform(df_data_space)
        # tsne784_2_df = pd.DataFrame(tsne784_2, columns=["tsne784_0", "tsne784_1"])
        # df = pd.concat([df, tsne784_2_df], axis=1)

        n_dim_list = [2] + list(range(4, 785, 60))
        n_dim_list = [2, 4, 14, 24, 34, 44, 54, 64, 94] + list(range(124, 785, 60))
        # n_dim_list = [2, 3, 4, 5, 48, 49, 50, 51]
        col_names = add_clustering_data(df, n_dim_list, dim_reduced_data)

        # add label col
        df = df.rename({label_col_name: "0"}, axis=1)
        col_names = ["0"] + (col_names)
        return df, col_names  # , None, "n_dim"

    elif which_data == "RegularTetrahedron_DBSCANDiffEps":
        a = 5  # side length
        mu = np.array(
            [
                [0, 0, 0],
                [a, 0, 0],
                [a / 2, a * np.sqrt(3) / 2, 0],
                [a / 2, a / (2 * np.sqrt(3)), np.sqrt(2 / 3) * a],
            ]
        )
        cov = 0.001 * np.identity(3)
        # Generate 1000 random samples from each Gaussian distribution
        samples = []
        np.random.seed(1)
        for i in range(4):
            samples.extend(np.random.multivariate_normal(mu[i], cov, 1000))
        samples = np.array(samples)
        df = pd.DataFrame(
            {
                "X": samples[:, 0],
                "Y": samples[:, 1],
                "Z": samples[:, 2],
            }
        )
        df_data_space = df[["X", "Y", "Z"]]
        # #  perform TSNE
        # tsne3_2 = TSNE(n_components=2, perplexity=500).fit_transform(df_data_space)
        # tsne3_2_df = pd.DataFrame(tsne3_2, columns=["tsne3_0", "tsne3_1"])
        # df = pd.concat([df, tsne3_2_df], axis=1)

        col_names = []
        min_samples = 10
        eps_list = [
            0.1,
            0.5,
            1,
            4,
            4.7,
            4.77,
            4.78,
            4.79,
            4.8,
            5,
            6,
        ]
        for eps in eps_list:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
                df_data_space.to_numpy()
            )
            col_name = str(eps)
            df[col_name] = clustering.labels_
            col_names.append(col_name)
        # column_details_df = pd.DataFrame({"ε": eps_list})
        column_details_df = pd.DataFrame({"ε": eps_list}, index=col_names)
        return df, col_names  # , eps_list, "ε"

    elif which_data == "3blobs2moons_KMeansDiff":
        col_names = []
        print("Computing sample data", end="... ")
        centers = [(-5, -5), (5, 5), (-5, 5)]
        cluster_std = [0.8, 1, 0.1]
        blobs = make_blobs(
            n_samples=1000,
            cluster_std=cluster_std,
            centers=centers,
            n_features=2,
            random_state=1,
        )
        moons = make_moons(n_samples=1000, noise=0.05, random_state=19)
        data_2d = np.concatenate((blobs[0], moons[0]), axis=0)
        x = data_2d[:, 0]
        y = data_2d[:, 1]
        df = pd.DataFrame({"x": x, "y": y})
        print("Done")

        print("Computing clustering ensembles", end="... ")
        os.environ["OMP_NUM_THREADS"] = "8"
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        n_list = list(range(3, 15 + 1, 1))
        for n in n_list:
            # Fit KMeans with 'n' clusters
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df[["x", "y"]])
            # Add the cluster labels as a new column
            col_name = f"{n}"
            df[col_name] = clusters
            col_names.append(col_name)
        column_details_df = pd.DataFrame({"n_clusters": n_list}, index=col_names)

        # column_details_df = pd.DataFrame({"n_clusters": n_list})
        print("Done")
        return df, col_names  # , n_list, "kmeans_n_clusters"

    elif which_data == "2blobs_KMeansDiff":
        col_names = []
        print("Computing sample data", end="... ")
        centers = [(-2, -2), (2, 2)]
        cluster_std = [1, 1]
        blobs = make_blobs(
            n_samples=10000,
            cluster_std=cluster_std,
            centers=centers,
            n_features=2,
            random_state=1,
        )
        data_2d = np.array(blobs[0])  # np.concatenate((blobs[0]), axis=0)
        x = data_2d[:, 0]
        y = data_2d[:, 1]
        df = pd.DataFrame({"x": x, "y": y})
        print("Done")

        print("Computing clustering ensembles", end="... ")
        os.environ["OMP_NUM_THREADS"] = "8"
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        n_list = range(2, 6 + 1, 1)
        for n in range(2, 6 + 1, 1):
            # Fit KMeans with 'n' clusters
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df[["x", "y"]])
            # Add the cluster labels as a new column
            col_name = f"{n}"
            df[col_name] = clusters
            col_names.append(col_name)
        print("Done")
        # column_details_df = pd.DataFrame({"n_clusters": n_list}, index=col_names)
        return df, col_names  # , None, "n_clusters"

    elif which_data == "RemainderFunction":
        # Define the number of rows and columns
        num_rows = 10000
        num_columns = 20

        # Initialize an empty dictionary to store the data
        data = {}
        col_names = []

        # Populate the dictionary with data based on the specified conditions
        for n in range(2, num_columns + 1):
            data[str(n)] = [i % n for i in range(num_rows)]
            col_names.append(str(n))
        # Create the DataFrame
        df = pd.DataFrame(data)
        column_details_df = pd.DataFrame(
            {"n": range(2, num_columns + 1)}, index=col_names
        )

        return df, col_names  # , None, "n_Remainder"

    elif which_data == "multi_label_classifier_results_22_to_30_epochs":
        df = read_csv_file("multi_label_classifier_epochs.csv")
        col_names = [str(i) for i in range(22, 31)]
        return df, col_names

    elif which_data == "contact_cliques_over_time":
        df = read_csv_file("InVS15_16clusters.csv")
        col_names = [
            "mon_1",
            "tue_1",
            "wed_1",
            "thu_1",
            "fri_1",
            "mon_2",
            "tue_2",
            "wed_2",
            "thu_2",
            "fri_2",
        ]
        return df, col_names

    else:
        print("Cannot find the specified dataset: " + str(which_data))
