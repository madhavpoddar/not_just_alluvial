import pandas as pd
import numpy as np
import networkx as nx
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from copy import deepcopy

# from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource


def calculate_dissimilarity(row_indexes, pair1, pair2):
    # Get the row indexes for the two pairs
    indexes1 = set(row_indexes.get(pair1, []))
    indexes2 = set(row_indexes.get(pair2, []))
    # Calculate the intersection and union of row indexes
    intersection = len(indexes1 & indexes2)
    union = len(indexes1 | indexes2)
    # Calculate dissimilarity using the formula
    dissimilarity = (
        1 - (intersection / union) if union > 0 else 1.0
    )  # Avoid division by zero
    return dissimilarity


def compute_dissimilarity_matrix(df, row_indexes):
    # Get the list of unique pairs of (Column Name, Unique Categorical Value)
    unique_pairs = list(row_indexes.keys())
    num_pairs = len(unique_pairs)
    # Initialize a dissimilarity matrix with zeros
    dissimilarity_matrix = np.zeros((num_pairs, num_pairs))
    # Calculate dissimilarity between all pairs
    for i in range(num_pairs):
        for j in range(i, num_pairs):
            pair1 = unique_pairs[i]
            pair2 = unique_pairs[j]
            dissimilarity = calculate_dissimilarity(row_indexes, pair1, pair2)
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity  # Since the matrix is symmetric
    return dissimilarity_matrix


def lab_to_rgb(a_star, b_star):
    L = 50  # Fixed value for L (lightness) at the mid-point
    # Create a LabColor object
    lab_color = LabColor(L, a_star * 256 - 128, b_star * 256 - 128)
    # Convert LabColor to sRGBColor
    rgb_color = convert_color(lab_color, sRGBColor)
    # Convert RGB values to 8-bit integers and format as a hex color
    r, g, b = (
        (np.clip(rgb_color.rgb_r, 0, 1) * 255).astype(int),
        (np.clip(rgb_color.rgb_g, 0, 1) * 255).astype(int),
        (np.clip(rgb_color.rgb_b, 0, 1) * 255).astype(int),
    )
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return hex_color


def two_opt(path, distance_matrix):
    """Improve the path using the 2-opt algorithm."""
    best_path = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                if j - i == 1:
                    continue  # Skip adjacent nodes
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                if calculate_total_distance(
                    new_path, distance_matrix
                ) < calculate_total_distance(best_path, distance_matrix):
                    best_path = new_path
                    improved = True
        path = best_path
    return best_path


def calculate_total_distance(path, distance_matrix):
    """Calculate the total distance of the TSP path."""
    total_distance = sum(
        distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)
    )
    return total_distance


def remove_longest_edge_from_cycle(cycle, dissimilarity_matrix):
    max_distance = -np.inf
    max_edge = None

    # Find the longest edge in the cycle
    for i in range(len(cycle) - 1):
        distance = dissimilarity_matrix[cycle[i]][cycle[i + 1]]
        if distance > max_distance:
            max_distance = distance
            max_edge = (cycle[i], cycle[i + 1])

    # Remove the longest edge to form a non-cycle path
    path = cycle[cycle.index(max_edge[1]) :] + cycle[1 : cycle.index(max_edge[0]) + 1]
    return path


def get_setwise_color_allocation(df):
    #####################################################################################################
    # Create a dictionary to store row indexes for each pair of (Column Name, Unique Categorical Value)
    #####################################################################################################
    row_indexes = {}
    # Iterate through the columns and unique values
    for col_name in df.columns:
        for unique_value in df[col_name].unique():
            key = (col_name, unique_value)
            row_indexes[key] = df[df[col_name] == unique_value].index.tolist()

    #####################################################################################################
    # Calculate the dissimilarity matrix (Jaccard index) and based on that the 2D MDS projection of sets
    #####################################################################################################
    dissimilarity_matrix = compute_dissimilarity_matrix(df, row_indexes)

    # Compute a 2D projection using MDS
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=19,
        normalized_stress="auto",
        max_iter=10000,
        eps=1e-9,
    )
    projection = mds.fit_transform(dissimilarity_matrix)

    # Calculate the pairwise Euclidean distances in the 2D projection
    new_dissimilarity_matrix = squareform(pdist(projection, metric="euclidean"))

    # Create a graph from the dissimilarity matrix
    G = nx.from_numpy_matrix(new_dissimilarity_matrix)
    # Solve the TSP using the approximate algorithm
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True)
    # Improve the initial TSP path using the 2-opt algorithm
    tsp_path = two_opt(tsp_path, dissimilarity_matrix)
    # Remove the longest edge to form a non-cycle path
    tsp_path = remove_longest_edge_from_cycle(tsp_path, new_dissimilarity_matrix)

    # Create a DataFrame for the projection data
    projection_df = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1]})
    projection_df["tsp_seq"] = projection_df.apply(
        lambda row: tsp_path.index(row.name), axis=1
    )
    print(projection_df)
    projection_df["label"] = list(row_indexes.keys())
    projection_df["partition_col_name"] = projection_df.apply(
        lambda row: row["label"][0], axis=1
    )
    projection_df["partition_set_categorical_value"] = projection_df.apply(
        lambda row: row["label"][1], axis=1
    )

    #####################################################################################################
    # 1D MDS projection of sets for sequencing of sets within individual partitions
    #####################################################################################################
    # # Compute a 1D projection using MDS
    # mds = MDS(
    #     n_components=1,
    #     dissimilarity="precomputed",
    #     random_state=19,
    #     normalized_stress="auto",
    # )
    # projection = mds.fit_transform(dissimilarity_matrix)
    # tsne = TSNE(n_components=1, random_state=19)
    # projection = tsne.fit_transform(projection)

    # # Create a PCA instance with one component
    # pca = PCA(n_components=1)
    # # Fit the PCA on the MDS projection
    # pca_component = pca.fit_transform(projection)

    # # Add the projection data to the projection df
    # projection_df["1D_proj"] = projection[:, 0]

    spacing_ratio = 0.5
    width_per_count = (1 - spacing_ratio) / len(df.index)
    projection_df_column_combined = None
    for col_name in df.columns:
        projection_df_column = deepcopy(
            projection_df[projection_df["partition_col_name"] == col_name].sort_values(
                "tsp_seq"
            )
        )
        projection_df_column = projection_df_column.reset_index(drop=True)
        if len(projection_df_column.index) == 1:
            spacing_width = 0
        else:
            spacing_width = spacing_ratio / (len(projection_df_column.index) - 1)
        projection_df_column["width"] = projection_df_column.apply(
            lambda row: (
                df[col_name].values == row["partition_set_categorical_value"]
            ).sum()
            * width_per_count,
            axis=1,
        )
        projection_df_column["y_end"] = (
            projection_df_column["width"].cumsum()
            + projection_df_column.index * spacing_width
        )
        projection_df_column["y_start"] = (
            projection_df_column["y_end"] - projection_df_column["width"]
        )
        if not isinstance(projection_df_column_combined, pd.DataFrame):
            projection_df_column_combined = projection_df_column
        else:
            projection_df_column_combined = pd.concat(
                [projection_df_column_combined, projection_df_column], ignore_index=True
            )
    projection_df = projection_df_column_combined

    #####################################################################################################
    # Allocate these points (sets) a color based on 2D CIELab (L value fixed)
    #####################################################################################################
    # Calculate centroid of the points
    centroid = (projection_df["x"].mean(), projection_df["y"].mean())
    # Calculate a* and b* values
    projection_df["a_star"] = projection_df["x"] - centroid[0]
    projection_df["b_star"] = projection_df["y"] - centroid[1]
    # Calculate angle from centroid to x-axis
    angles = np.arctan2(projection_df["b_star"], projection_df["a_star"])
    projection_df["angle"] = np.degrees(angles)
    # Normalize a* and b* values
    scaler = MinMaxScaler(feature_range=(0, 1))
    projection_df[["a_star", "b_star"]] = scaler.fit_transform(
        projection_df[["a_star", "b_star"]]
    )
    # Convert a*, b* to color representation
    projection_df["color"] = projection_df.apply(
        lambda row: lab_to_rgb(row["a_star"], row["b_star"]), axis=1
    )

    #####################################################################################################
    # Returning dict with key: (Column Name, Unique Categorical Value), value: color
    #####################################################################################################
    label_color_dict = dict(zip(projection_df["label"], projection_df["color"]))
    psets_vertical_ordering_df = projection_df
    return label_color_dict, psets_vertical_ordering_df


def get_color_sort_order(psets_color, selected_partition_col_name):
    color_sort_order = []
    for partition_col_name, unique_val in psets_color.keys():
        if partition_col_name == selected_partition_col_name:
            color_sort_order.append(psets_color[(partition_col_name, unique_val)])

    return color_sort_order
