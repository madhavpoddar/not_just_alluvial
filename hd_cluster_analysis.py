from bokeh.io import curdoc
import pandas as pd

from helper_functions_generic import read_csv_file
from sample_df import get_sample_df
from drcl_vis_wrapper import drcl_vis_wrapper

# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "RemainderFunction"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "2blobs_KMeansDiff"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "MNIST_DiffDim_KMeans25"
# )
# df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
#     "RegularTetrahedron_DBSCANDiffEps"
# )
df, col_names, col_names_as_list_of_numbers, sequential_variable_name = get_sample_df(
    "3blobs2moons_KMeansDiff"
)
drcl_vis = drcl_vis_wrapper(
    df, col_names, col_names_as_list_of_numbers, sequential_variable_name
)
curdoc().add_root(drcl_vis.layout)


# df = read_csv_file("InVS15_16clusters.csv")
# col_names = [
#     "mon_1",
#     "tue_1",
#     "wed_1",
#     "thu_1",
#     "fri_1",
#     "mon_2",
#     "tue_2",
#     "wed_2",
#     "thu_2",
#     "fri_2",
# ]
# drcl_vis = drcl_vis_wrapper(df, col_names)

# df = read_csv_file("InVS13_8clusters.csv")
# df = read_csv_file("SFHH_8clusters.csv")
# col_names = ["time_" + str(i) for i in range(len(df.columns) - 2)]
# col_names = ["clustering" + str(i) for i in range(len(df.columns) - 2)]
# drcl_vis = drcl_vis_wrapper(df, col_names)


# df = pd.DataFrame(
#     {
#         "0": [1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2],
#         "1": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
#         "2": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1],
#     }
# )
# df = pd.DataFrame(
#     {
#         "1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1],
#         "2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
#         "3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2],
#         "4": [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 3, 3, 3, 2, 2, 2, 2, 2, 2],
#         "5": [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2],
#         "6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 7, 8, 1, 1, 1, 1, 1, 1],
#     }
# )
# col_names = [str(i + 1) for i in range(len(df.columns))]
# drcl_vis = drcl_vis_wrapper(df, col_names, bool_pickle_preprocessing=False)

# df = read_csv_file("titanic_categorical.csv")
# col_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# drcl_vis = drcl_vis_wrapper(df, col_names)

### Dataset of authors regarding collaborations in publications on Visualization (VIS & predecessors) from 1990-2022 grouped by 5 year intervals.
### Dataset is filtered from 6600 authors to 36 by only including
### authors that collaborated in at least 3 of the time intervals (->102) with at least 2 other authors (->36).
# df = read_csv_file("vispub_filtered_102.csv")
# col_names = ['90-95', '95-00', '00-05', '05-10', '10-15', '15-20', '20-22']
# df = read_csv_file("vispub_filtered_74.csv")
# col_names = [str(i) for i in range(2000,2023)]
# df = read_csv_file("vispub_filtered_25.csv")
# col_names = [str(i) for i in range(2010, 2023)]
# drcl_vis = drcl_vis_wrapper(df, col_names)

# df = read_csv_file("multi_label_classifier_epochs.csv")
# col_names = [str(i) for i in range(22, 31)]
# drcl_vis = drcl_vis_wrapper(df, col_names)

# curdoc().add_root(drcl_vis.layout)
