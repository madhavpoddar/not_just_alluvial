from bokeh.io import curdoc

from helper_functions_generic import read_csv_file
from sample_df import get_sample_df
from drcl_vis_wrapper import drcl_vis_wrapper

df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
    "RemainderFunction"
)
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "2blobs_KMeansDiff"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "MNIST_DiffDim_KMeans25"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "RegularTetrahedron_DBSCANDiffEps"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "3blobs2moons_KMeansDiff"
# )

# drcl_vis = drcl_vis_wrapper(
#     df, col_names, col_names_as_list_of_numbers, sequential_variable_name
# )


# df = read_csv_file("InVS13_8clusters.csv")
# df = read_csv_file("InVS15_8clusters.csv")
df = read_csv_file("SFHH_8clusters.csv")
col_names = ["time_"+str(i) for i in range(len(df.columns)-2)]
drcl_vis = drcl_vis_wrapper(df, col_names)


curdoc().add_root(drcl_vis.layout)
