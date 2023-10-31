from bokeh.palettes import turbo
import random
import string
import glasbey


# all the visualization parameters are assigned here
def set_skewer_params(df, col_names):
    skewer_params = {}

    random.seed(42)
    # generating random string
    skewer_params["random_tag"] = str(
        "".join(
            random.choices(
                string.ascii_uppercase + string.ascii_lowercase + string.digits, k=11
            )
        )
    )

    # Width in screen pixels
    skewer_params["widthspx_alluvial"] = int(1120)
    skewer_params["widthspx_cim"] = skewer_params["widthspx_alluvial"]
    # skewer_params["widthspx_checkboxgrp_visible_cluster_cols"] = 150
    skewer_params["widthspx_mds_col_similarity_cl_membership"] = 350
    skewer_params["widthspx_ndimplot"] = skewer_params[
        "widthspx_mds_col_similarity_cl_membership"
    ]
    skewer_params["widthspx_similarity_roof_shaped_matrix_diagram"] = skewer_params[
        "widthspx_mds_col_similarity_cl_membership"
    ]
    skewer_params["widthspx_dissimilarity_data_table"] = 400

    # Height in screen pixels
    skewer_params["heightspx_rbg_edge_alpha_highlight"] = 30
    skewer_params["heightspx_alluvial"] = int(535)
    skewer_params["heightspx_cim"] = 120
    skewer_params["heightspx_ndimplot"] = skewer_params["widthspx_ndimplot"]
    skewer_params["heightspx_editdist"] = 400
    skewer_params["heightspx_mds_col_similarity_cl_membership"] = skewer_params[
        "widthspx_mds_col_similarity_cl_membership"
    ]
    skewer_params["heightspx_similarity_roof_shaped_matrix_diagram"] = (
        815 - skewer_params["heightspx_mds_col_similarity_cl_membership"]
    )
    skewer_params["heightspx_dissimilarity_data_table"] = 805

    skewer_params["alluvial_y_start"] = -0.1
    skewer_params["alluvial_y_end"] = 1.23
    skewer_params["pcp_width"] = 0.18

    skewer_params["spacing_ratio"] = 0.5

    skewer_params["pcp_ellipse_height"] = 0.02
    skewer_params["rb_ellipse_y"] = -0.06
    skewer_params["rb_labels_y"] = -0.07
    skewer_params["rb_ellipse_height"] = 0.05
    skewer_params["rb_ellipse_bondary_halfheight"] = (
        skewer_params["rb_ellipse_height"] * 0.57
    )
    skewer_params["rb_line_width"] = 2
    skewer_params["rb_hatch_pattern_filtered_column"] = "+"
    skewer_params["rb_hatch_pattern_not_filtered_column"] = " "

    # Colors
    skewer_params["cluster_bars_default_line_color"] = "black"
    skewer_params["cluster_bars_filtered_out_line_color"] = "gray"
    skewer_params["rb_fill_color_unselected"] = "white"
    skewer_params["rb_fill_color_selected"] = "black"
    skewer_params["rb_line_color"] = "lightgray"
    skewer_params["color_col_name"] = "drcl_color"

    # Dynamic Parameters
    skewer_params["bar_width"] = 0.003 * len(col_names)
    skewer_params["rb_ellipse_width"] = min(0.040 * len(col_names), 0.8)
    skewer_params["pcp_circle_radius"] = min(0.0077 * len(col_names), 0.1)
    skewer_params["rb_ellipse_bondary_halfwidth"] = (
        skewer_params["rb_ellipse_width"] * 0.57
    )
    skewer_params["cim_bar_width"] = min(
        skewer_params["rb_ellipse_bondary_halfwidth"] * 2, 0.98
    )
    skewer_params["width_per_count"] = (1 - skewer_params["spacing_ratio"]) / len(
        df.index
    )

    return skewer_params


def color_palette(num_colors: int, shuffle=True):
    return glasbey.create_palette(palette_size=num_colors)
    if num_colors == 2:
        return ["indigo", "goldenrod"]
    colors = list(turbo(num_colors))
    if shuffle:
        random.seed(0)
        random.shuffle(colors)
    return colors
