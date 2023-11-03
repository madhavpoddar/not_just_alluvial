import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

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


def get_setwise_color_allocation(df):
    #####################################################################################################
    # Create a dictionary to store row indexes for each pair of (Column Name, Unique Categorical Value)
    #####################################################################################################
    row_indexes = {}
    # Iterate through the columns and unique values
    for column_name in df.columns:
        for unique_value in df[column_name].unique():
            key = (column_name, unique_value)
            row_indexes[key] = df[df[column_name] == unique_value].index.tolist()

    #####################################################################################################
    # Calculate the dissimilarity matrix and based on that the 2D MDS projection of sets
    #####################################################################################################
    dissimilarity_matrix = compute_dissimilarity_matrix(df, row_indexes)
    # Compute a 2D projection using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed")
    projection = mds.fit_transform(dissimilarity_matrix)
    # Create a DataFrame for the projection data
    projection_df = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1]})
    projection_df["label"] = list(row_indexes.keys())

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
    print(projection_df[["a_star", "b_star"]])
    # Convert a*, b* to color representation
    projection_df["color"] = projection_df.apply(
        lambda row: lab_to_rgb(row["a_star"], row["b_star"]), axis=1
    )
    # print(projection_df)

    #####################################################################################################
    # Returning dict with key: (Column Name, Unique Categorical Value), value: color
    #####################################################################################################
    # projection_df = projection_df[["label", "color"]]
    label_color_dict = dict(zip(projection_df["label"], projection_df["color"]))
    return label_color_dict


def get_color_sort_order(
    psets_color_and_alluvial_position, selected_partition_col_name
):
    color_sort_order = []
    for partition_col_name, unique_val in psets_color_and_alluvial_position.keys():
        if partition_col_name == selected_partition_col_name:
            color_sort_order.append(
                psets_color_and_alluvial_position[(partition_col_name, unique_val)]
            )

    return color_sort_order


# # Sample DataFrame
# data = {
#     "Category_1": [
#         "a",
#         "b",
#         "c",
#         "a",
#         "b",
#         "c",
#         "b",
#         "c",
#         "c",
#         "b",
#         "a",
#         "a",
#         "b",
#         "b",
#         "a",
#     ],
#     "Category_2": [1, 3, 4, 3, 1, 4, 3, 1, 4, 1, 3, 4, 1, 3, 1],
#     "Category_3": [
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "b",
#     ],
# }
# df = pd.DataFrame(data)
# get_setwise_color_allocation(df)

# Create a Bokeh ColumnDataSource
# source = ColumnDataSource(projection_df)
# # Create a Bokeh plot
# p = figure(
#     title="2D Projection of Set Individual Partitions", width=800, height=600
# )
# p.scatter("x", "y", source=source, size=8, color="color")
# # # Add labels to the points
# # labels = LabelSet(x='x', y='y', text='label', level='glyph', x_offset=5, y_offset=5,
# #                   source=source, render_mode='canvas')
# # p.add_layout(labels)
# # Display the plot
# show(p)

# import pandas as pd
# import itertools

# # Create the dataframe and define col_names
# data = {
#     "Category_1": [
#         "a",
#         "b",
#         "c",
#         "a",
#         "b",
#         "c",
#         "b",
#         "c",
#         "c",
#         "b",
#         "a",
#         "a",
#         "b",
#         "b",
#         "a",
#     ],
#     "Category_2": [1, 3, 4, 3, 1, 4, 3, 1, 4, 1, 3, 4, 1, 3, 1],
#     "Category_3": [
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "d",
#         "b",
#         "c",
#         "b",
#     ],
# }
# df = pd.DataFrame(data)
# col_names = ["Category_1", "Category_2", "Category_3"]


# # Define a function to calculate the cost (crossings) of a given order
# def calculate_cost(order, df):
#     cost = 0
#     for i in range(len(order) - 1):
#         for j in range(i + 1, len(order)):
#             col_1, order_1 = order[i]
#             col_2, order_2 = order[j]
#             pairs = list(itertools.product(order_1, order_2))
#             cost += sum(
#                 df[df[col_1].eq(pair[0]) & df[col_2].eq(pair[1])].shape[0]
#                 for pair in pairs
#             )
#     return -cost  # We want to minimize the cost, so negate it


# # Initialize the order randomly
# initial_order = [(col, list(df[col].unique())) for col in col_names]

# # Import the simulated annealing optimizer from the SciPy library
# from scipy.optimize import basinhopping

# # Optimize the order
# result = basinhopping(calculate_cost, initial_order, minimizer_kwargs={"args": (df,)})
# final_order = {col: result.x[i][1] for i, col in enumerate(col_names)}

# print(final_order)
