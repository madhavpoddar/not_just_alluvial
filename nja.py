import pandas as pd
from bokeh.io import curdoc

from nja_sample_df import get_sample_df
from nja_wrapper import nja_wrapper


######################################################################################
# Set sample_dataset_name to None if you want to specify your own dataset
######################################################################################
# sample_dataset_name = None
sample_dataset_name = "3blobs2moons_KMeansDiff"

######################################################################################
# Custom data partition sequence:
######################################################################################
if sample_dataset_name in [
    "3blobs2moons_KMeansDiff",  # Figure 1
    "RemainderFunction",  # Figure 2 (but at a larger scale)
    "MNIST_DiffDim_KMeans25",  # Figure 5
    "contact_cliques_over_time",  # Figures 6, 7
    "multi_label_classifier_results_22_to_30_epochs",  # Figures 8, 9
    "RegularTetrahedron_DBSCANDiffEps",  # Not included in paper
]:
    df, col_names = get_sample_df(sample_dataset_name)
    nja = nja_wrapper(df, col_names)


######################################################################################
# Custom data partition sequence:
######################################################################################
elif sample_dataset_name == None:

    # The NJA wrapper takes 3 arguments:

    #
    # 1. df:
    #       pandas dataframe containing the data partition sequence
    df = pd.DataFrame(
        {
            "1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1],
            "2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2],
            "4": [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 3, 3, 3, 2, 2, 2, 2, 2, 2],
            "5": [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2],
            "6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 7, 8, 1, 1, 1, 1, 1, 1],
        }
    )

    #
    # 2. col_names:
    #       List of columns containing a partition each
    col_names = [str(i + 1) for i in range(len(df.columns))]

    #
    # 3. bool_pickle_preprocessing: (Optional argument set to True by default.)
    #       Boolean flag to indicate whether a pre-processing file should be generated or not.
    #       Pre-processing file generation stores preprocessing results for a dataset.
    #       Avoids running preprocessing every time the same dataset is visualized.
    #       Preferable to set it to False only for smaller datasets (in terms of # of sets).
    bool_pickle_preprocessing = False

    nja = nja_wrapper(df, col_names, bool_pickle_preprocessing)


curdoc().add_root(nja.layout)
