from bokeh.io import curdoc

from drcl_vis_wrapper import drcl_vis_wrapper
from sample_df import get_sample_df


# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "MNIST_DiffDim_KMeans25"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "RegularTetrahedron_DBSCANDiffEps"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "3blobs2moons_KMeansDiff"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "2blobs_KMeansDiff"
# )
df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
    "RemainderFunction"
)


# How does a sample input looks like for generating such visualization:
# (based on the remainder function example)
#
# import pandas as pd
# df = pd.DataFrame(
#     {
#         "numbers": [0, 1, 2, 3, 4, 5],
#         "2": [0, 1, 0, 1, 0, 1],
#         "3": [0, 1, 2, 0, 1, 2],
#         "4": [0, 1, 2, 3, 0, 1],
#     }
# )
# drcl_vis = drcl_vis_wrapper(
#     df=df,
#     col_names=["2", "3", "4"],                # of individual partition columns
#     col_names_as_list_of_numbers=[2, 3, 4],   # values corresponding to individual partition columns (optional)
#     sequential_variable_name="n_remainder",   # what these values signify (optional)
# )
#
# # Other example - including label partition
# drcl_vis = drcl_vis_wrapper(
#     df=some_other_df,
#     col_names=["2", "3", "4", "label"], # of individual partition columns
#     col_names_as_list_of_numbers=None,  # since the column names are not all numeric, they will be treated as categorical data
#     sequential_variable_name="n_dim",   # what these values signify (optional)
# )

drcl_vis = drcl_vis_wrapper(
    df, col_names, col_names_as_list_of_numbers, sequential_variable_name
)
curdoc().add_root(drcl_vis.layout)
