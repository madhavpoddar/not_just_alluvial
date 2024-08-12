import random
import string


# all the visualization parameters are assigned here
def set_nja_params(df, col_names):
    nja_params = {}

    random.seed(42)
    # generating random string
    nja_params["random_tag"] = str(
        "".join(
            random.choices(
                string.ascii_uppercase + string.ascii_lowercase + string.digits, k=11
            )
        )
    )

    # Width in screen pixels
    nja_params["widthspx_alluvial"] = int(1120)
    nja_params["widthspx_ixn_merge_split"] = nja_params["widthspx_alluvial"]
    nja_params["widthspx_nxn"] = 350

    # Height in screen pixels
    nja_params["heightspx_alluvial"] = int(535)
    nja_params["heightspx_ixn_merge_split"] = int(
        nja_params["heightspx_alluvial"] / 4.8
    )
    nja_params["heightspx_nxn"] = nja_params["widthspx_nxn"]

    nja_params["alluvial_y_start"] = -0.1
    nja_params["alluvial_y_end"] = 1.03
    # nja_params["pcp_width"] = 0.18

    nja_params["alluvial_spacing_ratio"] = 0.5

    # nja_params["pcp_ellipse_height"] = 0.02
    nja_params["rb_ellipse_y"] = -0.055
    nja_params["rb_labels_y"] = -0.0675
    nja_params["rb_ellipse_height"] = 0.06
    nja_params["rb_ellipse_bondary_halfheight"] = nja_params["rb_ellipse_height"] * 0.54
    nja_params["rb_line_width"] = 2
    nja_params["rb_hatch_pattern_filtered_column"] = "+"
    nja_params["rb_hatch_pattern_not_filtered_column"] = " "

    # Colors
    nja_params["cluster_bars_default_line_color"] = "black"
    nja_params["cluster_bars_filtered_out_line_color"] = "gray"
    nja_params["rb_fill_color_unselected"] = "white"
    nja_params["rb_fill_color_selected"] = "black"
    nja_params["rb_line_color"] = "lightgray"
    nja_params["color_col_name"] = "drcl_color"

    ########################################################################################
    # Dynamic Parameters (Depends on number of partitions / # of data items / etc.)
    ########################################################################################

    nja_params["bar_width"] = 0.006 * len(col_names)
    nja_params["rb_ellipse_width"] = min(
        0.055 * len(col_names), 0.9
    )  # old: min(0.040 * len(col_names), 0.8)
    nja_params["pcp_circle_radius"] = min(0.0077 * len(col_names), 0.1)
    nja_params["rb_ellipse_bondary_halfwidth"] = nja_params["rb_ellipse_width"] * 0.54
    nja_params["cim_bar_width"] = min(
        nja_params["rb_ellipse_bondary_halfwidth"] * 1.6, 0.9
    )
    nja_params["width_per_count"] = (1 - nja_params["alluvial_spacing_ratio"]) / len(
        df.index
    )

    return nja_params
