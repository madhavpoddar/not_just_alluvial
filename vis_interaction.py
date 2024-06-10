import copy
from helper_functions_project_specific import get_unique_vals
from vis_params import color_palette
from helper_functions_generic import timer


def set_initial_curr_selection(col_names):
    return {
        "color_col_name": col_names[0],  # [col_names[0]],
        "cluster_ids": [],
        "ndimplot_col_names": [],
        # "scatterplot_col_names": {"x": "pca_0", "y": "pca_1"},
    }


def calc_index_cid_clicked(event, col_names, psets_vertical_ordering_df):
    index = round(event.x)
    if index in range(len(col_names)):
        c_id_index = -1
        col_name_filtered_setwise_position_df = psets_vertical_ordering_df[
            psets_vertical_ordering_df["partition_col_name"] == col_names[index]
        ]
        y_starts = col_name_filtered_setwise_position_df["y_start"].values
        for i in range(len(y_starts)):
            if i == len(y_starts) - 1:
                if event.y > y_starts[i]:
                    c_id_index = i
                    break
            else:
                if event.y > y_starts[i] and event.y < y_starts[i + 1]:
                    c_id_index = i
                    break
        if c_id_index != -1:
            return (
                index,
                col_name_filtered_setwise_position_df[
                    "partition_set_categorical_value"
                ].iloc[i],
            )
    return None, None


def calc_curr_selection(event, old_selection, psets_vertical_ordering_df, col_names):
    curr_selection = copy.deepcopy(old_selection)
    if event.y < 0:
        # Unselect existing curr_selection circle
        cs_col_name = col_names[round(event.x)]

        curr_selection["cluster_ids"] = []
        if (old_selection["color_col_name"] != cs_col_name) or (
            len(old_selection["cluster_ids"]) != 0
        ):
            curr_selection["color_col_name"] = cs_col_name
        else:
            curr_selection["color_col_name"] = None
    else:
        cs_col_id, cs_cluster_id = calc_index_cid_clicked(
            event, col_names, psets_vertical_ordering_df
        )
        if cs_col_id == None:
            return curr_selection
        cs_col_name = col_names[cs_col_id]
        if old_selection["color_col_name"] != cs_col_name:
            curr_selection["color_col_name"] = cs_col_name
            curr_selection["cluster_ids"] = [cs_cluster_id]
        else:
            if cs_cluster_id not in curr_selection["cluster_ids"]:
                curr_selection["cluster_ids"].append(cs_cluster_id)
                curr_selection["cluster_ids"].sort()
            else:
                curr_selection["cluster_ids"].remove(cs_cluster_id)
                if len(curr_selection["cluster_ids"]) == 0:
                    curr_selection["color_col_name"] = None
    return curr_selection


def df_assign_colors(
    df,
    psets_color,
    selected_partition_col_name,
    color_col_name,
    remove_colors=False,
):
    if remove_colors:
        df.loc[:, color_col_name] = "gray"
    else:
        # unique_vals = get_unique_vals(df, selected_partition_col_name)
        # print(psets_color)
        # for unique_val in unique_vals:
        #     print(
        #         psets_color[
        #             (selected_partition_col_name, unique_val)
        #         ]
        #     )
        # colors = color_palette(min(len(unique_vals), 256))
        # df[color_col_name] = df.apply(
        #     lambda row: colors[
        #         unique_vals.index(row[selected_partition_col_name]) % 256
        #     ],
        #     axis=1,
        # )

        df[color_col_name] = df.apply(
            lambda row: psets_color[
                (selected_partition_col_name, row[selected_partition_col_name])
            ],
            axis=1,
        )


def calc_df_filtered(df, curr_selection):
    if (
        curr_selection["color_col_name"] != None
        and len(curr_selection["cluster_ids"]) != 0
    ):
        df_filtered = df[
            df[curr_selection["color_col_name"]].isin(curr_selection["cluster_ids"])
        ]
    else:
        df_filtered = df
    return df_filtered


def selection_update_tap(
    curr_selection,
    old_selection,
    df,
    fig_obj,
    col_names,
    psets_vertical_ordering_df,
    psets_color,
    skewer_params,
):
    # assign colors based on unique values of color_col_name
    df_assign_colors(
        df,
        psets_color,
        curr_selection["color_col_name"],
        skewer_params["color_col_name"],
        remove_colors=len(curr_selection["cluster_ids"]) == 0
        and curr_selection["color_col_name"] == None,
    )
    df_filtered = calc_df_filtered(df, curr_selection)

    fig_obj["alluvial"].update_selection(
        df,
        df_filtered,
        psets_vertical_ordering_df,
        psets_color,
        skewer_params,
        col_names,
        curr_selection,
        old_selection,
    )
    fig_obj["cim"].update_selection(
        df,
        df_filtered,
        skewer_params,
        psets_color,
        col_names,
        curr_selection,
        old_selection,
    )
    fig_obj["mds_nxn_setwise"].update_selection(
        df,
        df_filtered,
        skewer_params,
        col_names,
        curr_selection,
        psets_vertical_ordering_df,
    )
    # fig_obj["metamap_edit_dist"].update_selection(
    #     df_filtered, skewer_params, col_names, curr_selection, old_selection
    # )
    # fig_obj["ndimplot"].update_selection(
    #     df_filtered, skewer_params, col_names, curr_selection, old_selection
    # )
    return df, df_filtered


# def selection_update_tap(
#     curr_selection,
#     old_selection,
#     df,
#     fig_obj,
#     col_names,
#     psets_vertical_ordering_df,
#     skewer_params,
# ):
#     selection_update_double_tap(
#         curr_selection,
#         old_selection,
#         df,
#         fig_obj,
#         col_names,
#         psets_vertical_ordering_df,
#         skewer_params,
#     )
