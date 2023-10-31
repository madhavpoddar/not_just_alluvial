import numpy as np

from helper_functions_project_specific import get_unique_vals
from helper_functions_project_specific import get_cluster_count


def get_count_0_1(df, col_names):
    count_0_1 = {}
    subs_count = 0
    for col_name_0, col_name_1 in col_names.get_neighbouring_pairs_l2r():
        cluster_ids_0 = get_unique_vals(df, col_name_0)
        cluster_ids_1 = get_unique_vals(df, col_name_1)
        cluster_count_0 = len(cluster_ids_0)
        cluster_count_1 = len(cluster_ids_1)

        count_0_1[(col_name_0, col_name_1)] = np.zeros(
            (cluster_count_0, cluster_count_1)
        )

        for i, cluster_id_0 in enumerate(cluster_ids_0):
            for j, cluster_id_1 in enumerate(cluster_ids_1):
                subs_count += 1
                count_0_1[(col_name_0, col_name_1)][i][j] = df[
                    (df[col_name_0] == cluster_id_0) & (df[col_name_1] == cluster_id_1)
                ].shape[0]
    return count_0_1, subs_count


def modify_cluster_ids(df, col_name, old_cids, new_cids):
    dummy_constant = len(old_cids) + 10
    for old_id, new_id in zip(old_cids, new_cids):
        df = df.replace({col_name: old_id}, new_id + dummy_constant)
    df[col_name] = df[col_name] - dummy_constant
    return df


def reduce_intersections_neighbours(df, col_names):
    for (
        col_name_0,
        col_name_1,
    ) in col_names.get_neighbouring_pairs_l2r():
        cluster_ids_0 = get_unique_vals(df, col_name_0)
        cluster_ids_1 = get_unique_vals(df, col_name_1)
        cluster_count_0 = len(cluster_ids_0)
        cluster_count_1 = len(cluster_ids_1)
        max_counts_cluster_index_1 = np.zeros(cluster_count_0)

        count_0_1 = np.zeros((cluster_count_0, cluster_count_1))

        for i, cluster_id_0 in enumerate(cluster_ids_0):
            for j, cluster_id_1 in enumerate(cluster_ids_1):
                count_0_1[i][j] = df[
                    (df[col_name_0] == cluster_id_0) & (df[col_name_1] == cluster_id_1)
                ].shape[0]

        for i in range(cluster_count_0):
            max_counts_cluster_index_1[i] = np.argmax(count_0_1[i])
        cluster_ids_0_custom_sort = [
            x for _, x in sorted(zip(max_counts_cluster_index_1, cluster_ids_0))
        ]

        # re-assigning new Cluster IDs
        df = modify_cluster_ids(
            df=df,
            col_name=col_name_0,
            old_cids=cluster_ids_0_custom_sort,
            new_cids=cluster_ids_0,
        )

    return df


def calc_FMI(df, col_name_0, col_name_1, return_type_df=True):
    if col_name_0 == col_name_1:
        if return_type_df:
            df_FMI = df.groupby([col_name_0, col_name_1]).size().to_frame("FMI")
            df["FMI"] = 0
        return 0
    df_FMI = df.groupby([col_name_0, col_name_1]).size().to_frame("TP")
    df_FMI_temp = df_FMI.groupby(level=0).sum().rename(columns={"TP": "FP"})
    df_FMI = df_FMI.join(df_FMI_temp.reindex(df_FMI.index, level=0))
    df_FMI["FP"] = df_FMI["FP"] - df_FMI["TP"]
    df_FMI_temp = df_FMI[["TP"]].groupby(level=1).sum().rename(columns={"TP": "FN"})
    df_FMI = df_FMI.join(df_FMI_temp)
    df_FMI["FN"] = df_FMI["FN"] - df_FMI["TP"]
    df_FMI["FMI"] = df_FMI["TP"] / np.sqrt(
        (df_FMI["TP"] + df_FMI["FP"]) * (df_FMI["TP"] + df_FMI["FN"])
    )
    if return_type_df:
        return df_FMI["FMI"]
    return (df_FMI["FMI"] * df_FMI["TP"] / df_FMI["TP"].sum()).sum()


def calc_FMI_matrix(df, col_names):
    FMI_matrix = np.empty(shape=(len(col_names), len(col_names)))
    for i, col_name_0 in enumerate(col_names):
        for j, col_name_1 in enumerate(col_names):
            if i < j:
                FMI_matrix[i][j] = calc_FMI(
                    df, col_name_0, col_name_1, return_type_df=False
                )
            elif i == j:
                FMI_matrix[i][j] = 1
            else:
                FMI_matrix[i][j] = FMI_matrix[j][i]
    return FMI_matrix


def calc_alluvial_bar_params(df, col_names, spacing_ratio=0.5):
    count_0_1, alluvial_edges_count = get_count_0_1(df, col_names)

    width_per_count = (1 - spacing_ratio) / len(df.index)

    y_start = [None] * len(col_names)
    width = [None] * len(col_names)
    for col_id, col_name in enumerate(col_names):
        cluster_ids = get_unique_vals(df, col_name)
        if (len(cluster_ids)) == 1:
            spacing_width = 0
            starting_y = 0
        else:
            spacing_width = spacing_ratio / (len(cluster_ids) - 1)
            starting_y = 0
        y_start[col_id] = [None] * len(cluster_ids)
        width[col_id] = [None] * len(cluster_ids)

        for i, c_id in enumerate(cluster_ids):
            y_start[col_id][i] = starting_y
            width[col_id][i] = get_cluster_count(df, col_name, c_id) * width_per_count
            starting_y += width[col_id][i] + spacing_width

    return (
        count_0_1,
        y_start,
    )
