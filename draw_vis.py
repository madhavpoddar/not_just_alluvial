import numpy as np
import pandas as pd
import copy
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.manifold import MDS
from bokeh.plotting import figure
from bokeh.models import (
    Span,
    HoverTool,
    LabelSet,
    ColumnDataSource,
    Select,
    Button,
    MultiChoice,
    MultiSelect,
    CheckboxGroup,
    CheckboxButtonGroup,
    RadioButtonGroup,
    DataTable,
    TableColumn,
    BoxZoomTool,
    PanTool,
    WheelZoomTool,
    ResetTool,
    LassoSelectTool,
    Circle,
    LinearAxis,
    BoxAnnotation,
)
import random
from vis_params import color_palette
from df_preprocessing import calc_FMI
from helper_functions_project_specific import to_empty_dict, get_unique_vals
from helper_functions_generic import timer
from color_mapping_sets import get_color_sort_order


def draw_vlines(p, col_names):
    for i in range(len(col_names)):
        vline = Span(
            location=i,
            dimension="height",
            line_color="lightgray",
            line_width=1,
            level="underlay",
        )
        p.renderers.extend([vline])


def compute_edit_distance(str1, str2):
    edit_distance = 0
    for i, c1 in enumerate(str1):
        if c1 != str2[i]:
            edit_distance += 1
    return edit_distance


def number_to_text_bar_chart(num_bw_0_1, n_char):
    # https://en.wikipedia.org/wiki/Block_Elements
    # block_elements = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    block_elements = [" ", "░", "▒", "▓", "█"]
    num = num_bw_0_1 * n_char
    str_bar = block_elements[-1] * int(num)
    num = int((num - int(num)) * (len(block_elements) - 1))
    if num > 0:
        str_bar += block_elements[num]
    return str_bar


def smooth_sigmoid(x, x1, x2):
    return x1 + (x2 - x1) / (1 + np.exp(-x))


def pcp_interpolate_values_x(x_list, N=10000, n=10):
    assert 2 * n < N, "n should be such that 2 * n < N"
    interpolated_array = []

    for i in range(len(x_list) - 1):
        x1, x2 = x_list[i], x_list[i + 1]

        x_non_smooth = np.concatenate([[x1] * n, [x2] * n])
        y_non_smooth = np.concatenate(
            [
                np.linspace(0, 0.2, n),
                np.linspace(0.8, 1, n),
            ]
        )
        spl = InterpolatedUnivariateSpline(y_non_smooth, x_non_smooth)
        y_smooth = np.linspace(0, 1, N)
        x_smooth = list(spl(y_smooth))
        interpolated_array.extend(x_smooth)

        # # Repeating x1 n times
        # interpolated_array.extend([x1] * n)
        # # Interpolating between x1 and x2 in a curved manner (using sine function)
        # print(x1,x2)
        # print(smooth_sigmoid(np.linspace(0, 1, 10), x1, x2))
        # interpolated_array.append(smooth_sigmoid(np.linspace(0, 1, N-2*n), x1, x2))
        # # Repeating x2 n times
        # interpolated_array.extend([x2] * n)

    # Add the last element of x_list
    interpolated_array.extend([x_list[-1]] * n)
    # x_non_smooth = np.concatenate(
    #     [
    #         np.linspace(x0, x0 + (x1 - x0) * 0.1, 10),
    #         np.linspace(x0 + (x1 - x0) * 0.9, x1, 10),
    #     ]
    # )
    # y_non_smooth = np.concatenate([[y0_start] * 10, [y1_start] * 10])
    # spl = InterpolatedUnivariateSpline(x_non_smooth, y_non_smooth)
    # x_smooth = np.linspace(x0, x1, self.resolution)
    # y1_smooth = spl(x_smooth)
    # y2_smooth = y1_smooth + width
    return interpolated_array


def pcp_interpolate_values_y(y_list, N=10000):
    interpolated_array = []
    for i in range(len(y_list) - 1):
        y1, y2 = y_list[i], y_list[i + 1]
        step = (y2 - y1) / N
        for j in range(N):
            interpolated_array.append(y1 + step * j)
    interpolated_array.append(y_list[-1])  # Add the last value of y_list
    return interpolated_array


def calculate_tick_values(data, num_ticks=5):
    min_val = min(data)
    max_val = max(data)

    # # Calculate the range of the data
    # data_range = max_val - min_val
    # if data_range == 0:
    #     data_range = data[0]

    # # Determine the order of magnitude for the data range
    # order_of_magnitude = int(np.floor(np.log10(data_range)))

    # # Calculate the base tick interval using the order of magnitude
    # base_tick_interval = 10 ** order_of_magnitude

    # # Calculate the rounded tick interval to get rounded tick values
    # rounded_tick_interval = round(data_range / (num_ticks - 1), -order_of_magnitude)

    # # Use the maximum of base_tick_interval and rounded_tick_interval to ensure ticks are not too sparse
    # tick_interval = max(base_tick_interval, rounded_tick_interval)

    # # Calculate the starting point for the tick values to ensure they are within the data range
    # start_value = np.floor(min_val / tick_interval) * tick_interval

    # # Generate the tick values
    # tick_values = np.arange(start_value, max_val + tick_interval, tick_interval)

    # return tick_values

    if min_val != max_val:
        return [min_val, max_val]
    return [data[0] - 1, data[0] + 1]


# Custom function to concatenate lists and remove the first element
def concat_and_remove_first(lst):
    concatenated_list = sum(lst, [])  # Concatenate lists
    return concatenated_list[1:]  # Remove the first element


def group_radio_circles(x, y, threshold):
    result_x_single = []
    result_y_single = []
    result_x_multiple = []
    result_y_multiple = []

    group_x = [x[0]]
    group_y = [y[0]]

    for i in range(1, len(x)):
        if x[i] - x[i - 1] < threshold:
            group_x.append(x[i])
            group_y.append(y[i])
        else:
            if len(group_x) == 1:
                result_x_single.append(group_x[0])
                result_y_single.append(group_y[0])
            else:
                result_x_multiple.append(group_x)
                result_y_multiple.append(group_y)
            group_x = [x[i]]
            group_y = [y[i]]

    # Add the last group
    if len(group_x) == 1:
        result_x_single.append(group_x[0])
        result_y_single.append(group_y[0])
    else:
        result_x_multiple.append(group_x)
        result_y_multiple.append(group_y)

    return result_x_single, result_y_single, result_x_multiple, result_y_multiple


def generate_continuous_border(x1, y1, x2, y2, w, h):
    # Create an array of angles for the ellipse
    angles = np.linspace(-np.pi * 0.5, np.pi * 0.5, 200)

    # Calculate the x and y coordinates of the ellipse
    x = np.concatenate([x1 - w * np.cos(angles), x2 + w * np.cos(angles[::-1])])
    y = np.concatenate([y1 + h * np.sin(angles), y2 + h * np.sin(angles[::-1])])

    return list(x), list(y)


def groupped_circles_to_patches(groupped_circles_x, groupped_circles_y, w, h):
    patches_xs = []
    patches_ys = []
    for x_list, y_list in zip(groupped_circles_x, groupped_circles_y):
        patch_xs, patch_ys = generate_continuous_border(
            x_list[0], y_list[0], x_list[-1], y_list[-1], w, h
        )
        patches_xs.append(patch_xs)
        patches_ys.append(patch_ys)

    return patches_xs, patches_ys


def reduce_pd_series_to_n_values_preserving_freq_dist(df_series, n=1000):
    # Calculate the frequency of each unique value
    value_counts = df_series.value_counts()

    # Calculate the total count of unique values
    total_counts = value_counts.sum()

    # Calculate the ratio to distribute counts proportionally
    ratios = (value_counts / total_counts).sort_index()

    # Calculate the number of occurrences you want for each unique value
    desired_counts = (ratios * n).astype(int)

    # Create the reduced list
    reduced_list = []
    for value, count in desired_counts.items():
        reduced_list.extend([value] * count)

    # If the reduced list length is less than the desired length, you can fill it with additional values
    remaining_count = n - len(reduced_list)
    if remaining_count > 0:
        additional_values = list(
            [value_counts.keys()[-1]] * remaining_count
        )  # [-(1 + remaining_count) : -1]
        reduced_list.extend(additional_values)

    # Now, reduced_list contains the desired list of 1000 elements preserving the ratio of frequencies
    return reduced_list


class alluvial:
    def __init__(
        self,
        df,
        column_details_df,
        psets_vertical_ordering_df,
        psets_color,
        skewer_params,
        col_names,
        curr_selection,
    ):
        self.p = self.generate_figure(skewer_params)
        draw_vlines(self.p, col_names)
        timer_obj = timer("Updating Alluvial Diagram")
        self.alluvial_cluster_bars_obj = self.alluvial_cluster_bars(
            self.p,
            df,
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
        )
        self.alluvial_edges_obj = self.alluvial_edges(
            self.p,
            df,
            df,
            psets_vertical_ordering_df,
            psets_color,
            skewer_params,
            col_names,
            curr_selection,
        )
        self.rbg_edge_alpha_highlight = RadioButtonGroup(
            labels=["Highlight Inconsistency", "No Highlight", "Highlight Consistency"],
            active=self.alluvial_edges_obj.rbg_edge_alpha_highlight_active,
            height=skewer_params["heightspx_rbg_edge_alpha_highlight"],
            width=skewer_params["widthspx_alluvial"],
        )
        self.rb_obj = self.radio_buttons(
            self.p,
            skewer_params,
            col_names,
            curr_selection,
        )

        self.number_line_df = self.add_multi_level_number_line(
            column_details_df, skewer_params, col_names, self.p
        )

        self.number_line_selection_ellipse_obj = self.number_line_selection_ellipse(
            self.p,
            df,
            df,
            skewer_params,
            col_names,
            curr_selection,
            self.number_line_df,
        )

        timer_obj.done()

    def add_multi_level_number_line(
        self, column_details_df, skewer_params, col_names, p
    ):
        variables = list(column_details_df.columns)
        column_details_df = copy.deepcopy(column_details_df)
        self.p.quad(
            left=0,
            right=len(column_details_df.index) - 1,
            bottom=1.02,
            top=2,
            color="white",
            line_color="white",
            line_width=2,
        )
        dict_np1d_ticks = {}
        axis_locations_y = {}
        for i, var in enumerate(variables):
            axis_locations_y[var] = 1.02 + i * 0.18 / (
                (len(variables) - 1) if len(variables) > 1 else 1
            )
            dict_np1d_ticks[var] = calculate_tick_values(list(column_details_df[var]))
            if var == skewer_params["random_tag"]:
                continue
            axis = Span(
                location=axis_locations_y[var],
                dimension="width",
                line_color="lightgray",
            )
            self.p.add_layout(axis)

        # Set the custom labels for specific tick positions
        axis_locations_y_without_starting_line = copy.deepcopy(axis_locations_y)
        del axis_locations_y_without_starting_line[skewer_params["random_tag"]]
        self.p.yaxis.ticker = list(axis_locations_y_without_starting_line.values())
        self.p.yaxis.major_label_overrides = dict(
            zip(
                list(axis_locations_y_without_starting_line.values()),
                list(axis_locations_y_without_starting_line.keys()),
            )
        )

        list_ys = [axis_locations_y[var] for var in variables]
        column_details_df[
            skewer_params["random_tag"] + "_xs_straight"
        ] = column_details_df.apply(
            lambda row: [
                np.interp(
                    row[var],
                    [dict_np1d_ticks[var][0], dict_np1d_ticks[var][-1]],
                    [0, len(col_names) - 1],
                )
                for var in variables
            ],
            axis=1,
        )
        column_details_df[skewer_params["random_tag"] + "_ys_straight"] = [
            list_ys for i in column_details_df.index
        ]
        column_details_df[
            skewer_params["random_tag"] + "_xs"
        ] = column_details_df.apply(
            lambda row: pcp_interpolate_values_x(
                row[skewer_params["random_tag"] + "_xs_straight"]
            ),
            axis=1,
        )
        column_details_df[
            skewer_params["random_tag"] + "_ys"
        ] = column_details_df.apply(
            lambda row: pcp_interpolate_values_y(
                row[skewer_params["random_tag"] + "_ys_straight"]
            ),
            axis=1,
        )
        source = ColumnDataSource(column_details_df)
        self.p.multi_line(
            source=source,
            xs=skewer_params["random_tag"] + "_xs",
            ys=skewer_params["random_tag"] + "_ys",
            line_color="lightgray",
        )
        # Apply the custom function to the 'xs' column and store the result as a list
        column_details_df[
            skewer_params["random_tag"] + "_nl_ellipse_center_xs"
        ] = column_details_df.apply(
            lambda row: row[skewer_params["random_tag"] + "_xs_straight"][1], axis=1
        )
        circle_xs = column_details_df[
            skewer_params["random_tag"] + "_nl_ellipse_center_xs"
        ].to_list()
        column_details_df[
            skewer_params["random_tag"] + "_nl_ellipse_center_ys"
        ] = column_details_df.apply(
            lambda row: row[skewer_params["random_tag"] + "_ys_straight"][1], axis=1
        )
        circle_ys = column_details_df[
            skewer_params["random_tag"] + "_nl_ellipse_center_ys"
        ].to_list()

        # For multi_level_number_line change this to:
        # column_details_df[
        #     skewer_params["random_tag"] + "_nl_ellipse_center_xs"
        # ] = column_details_df.apply(
        #     lambda row: row[skewer_params["random_tag"] + "_xs_straight"][1:], axis=1
        # )
        # circle_xs = column_details_df.explode(
        #     skewer_params["random_tag"] + "_nl_ellipse_center_xs"
        # )[skewer_params["random_tag"] + "_nl_ellipse_center_xs"].to_list()
        # column_details_df[
        #     skewer_params["random_tag"] + "_nl_ellipse_center_ys"
        # ] = column_details_df.apply(
        #     lambda row: row[skewer_params["random_tag"] + "_ys_straight"][1:], axis=1
        # )
        # circle_ys = column_details_df.explode(
        #     skewer_params["random_tag"] + "_nl_ellipse_center_ys"
        # )[skewer_params["random_tag"] + "_nl_ellipse_center_ys"].to_list()

        (
            circle_xs,
            circle_ys,
            circle_xs_groupped,
            circle_ys_groupped,
        ) = group_radio_circles(
            circle_xs, circle_ys, threshold=skewer_params["pcp_circle_radius"] * 2
        )

        circles_df = pd.DataFrame({"x": circle_xs, "y": circle_ys})
        circles_cds = ColumnDataSource(circles_df)
        self.p.ellipse(
            source=circles_cds,
            x="x",
            y="y",
            fill_color="white",
            line_color="black",
            width=skewer_params["pcp_circle_radius"] * 2,
            height=skewer_params["pcp_ellipse_height"] * 2,
            level="overlay",
        )
        patches_xs, patches_ys = groupped_circles_to_patches(
            circle_xs_groupped,
            circle_ys_groupped,
            w=skewer_params["pcp_circle_radius"],
            h=skewer_params["pcp_ellipse_height"],
        )
        self.p.patches(
            patches_xs,
            patches_ys,
            fill_alpha=1,
            fill_color="white",
            line_color="black",
            line_width=1,
            level="overlay",
        )

        return column_details_df[
            [
                skewer_params["random_tag"] + "_nl_ellipse_center_xs",
                skewer_params["random_tag"] + "_nl_ellipse_center_ys",
            ]
        ]

    class number_line_selection_ellipse:
        def __init__(
            self,
            p,
            df,
            df_filtered,
            skewer_params,
            col_names,
            curr_selection,
            number_line_df,
        ):
            self.resolution = 512
            self.glyph_vars = ["xs", "ys", "fill_color", "alpha"]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.patches(
                source=src,
                xs="xs",
                ys="ys",
                fill_color="fill_color",
                line_color=None,
                line_width=0,
                alpha="alpha",
                level="overlay",
            )
            self.update_selection(
                number_line_df,
                df,
                df_filtered,
                skewer_params,
                col_names,
                curr_selection,
            )

        def get_cds_dict(
            self,
            number_line_df,
            df,
            skewer_params,
            col_names,
            curr_selection,
        ):
            if curr_selection["color_col_name"] == None:
                return to_empty_dict(self.glyph_vars)
            center_x = number_line_df.loc[curr_selection["color_col_name"]][
                skewer_params["random_tag"] + "_nl_ellipse_center_xs"
            ]
            center_y = number_line_df.loc[curr_selection["color_col_name"]][
                skewer_params["random_tag"] + "_nl_ellipse_center_ys"
            ]
            angle_linspace = np.linspace(
                0 + np.pi / 2, 2 * np.pi + np.pi / 2, num=self.resolution + 1
            )
            df_number_line_selection_ellipse_triangles = pd.DataFrame(
                list(zip(angle_linspace[:-1], angle_linspace[1:])),
                columns=["angle0", "angle1"],
            )
            df_number_line_selection_ellipse_triangles[
                "x0"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_circle_radius"]
                * np.cos(row["angle0"])
                * 0.8
                + center_x,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "x1"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_circle_radius"]
                * np.cos(row["angle1"])
                * 0.8
                + center_x,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "x2"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_circle_radius"]
                * np.cos((row["angle0"] + row["angle1"]) / 2)
                * 0.15
                + center_x,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "y0"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_ellipse_height"]
                * np.sin(row["angle0"])
                * 0.75
                + center_y,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "y1"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_ellipse_height"]
                * np.sin(row["angle1"])
                * 0.75
                + center_y,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "y2"
            ] = df_number_line_selection_ellipse_triangles.apply(
                lambda row: skewer_params["pcp_ellipse_height"]
                * np.sin((row["angle0"] + row["angle1"]) / 2)
                * 0.15
                + center_y,
                axis=1,
            )
            df_number_line_selection_ellipse_triangles[
                "xs"
            ] = df_number_line_selection_ellipse_triangles[
                ["x0", "x1", "x2"]
            ].values.tolist()
            df_number_line_selection_ellipse_triangles[
                "ys"
            ] = df_number_line_selection_ellipse_triangles[
                ["y0", "y1", "y2"]
            ].values.tolist()
            df_number_line_selection_ellipse_triangles[
                "fill_color"
            ] = reduce_pd_series_to_n_values_preserving_freq_dist(
                df["drcl_color"], n=self.resolution
            )
            df_number_line_selection_ellipse_triangles["alpha"] = 1

            return df_number_line_selection_ellipse_triangles[self.glyph_vars].to_dict(
                "list"
            )

        def update_selection(
            self,
            number_line_df,
            df,
            df_filtered,
            skewer_params,
            col_names,
            curr_selection,
            old_selection=None,
        ):
            self.glyph.data_source.data = self.get_cds_dict(
                number_line_df,
                df,
                skewer_params,
                col_names,
                curr_selection,
            )

    class vis_pcp:
        def __init__(self, p, df, skewer_params, col_names, curr_selection):
            self.glyphs = {}
            self.glyph_vars = {}
            self.glyph_vars["axes"] = ["x", "y", "color", "col_name"]

            # self.glyphs["axes"] =

            pass

        def generate_figure(self, skewer_params):
            p = figure(
                width=skewer_params["widthspx_mds_col_similarity_cl_membership"],
                height=skewer_params["heightspx_mds_col_similarity_cl_membership"],
                tools="",
            )
            p.toolbar.logo = None
            p.min_border = 0
            return p

    class alluvial_cluster_bars:
        def __init__(
            self,
            p,
            df,
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
        ):
            self.glyph_vars = [
                "left",
                "right",
                "top",
                "bottom",
                "line_color",
                "fill_color",
                "fill_alpha",
            ]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.quad(
                source=src,
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                line_width=1,
                line_color="line_color",
                fill_color="fill_color",
                fill_alpha="fill_alpha",
                level="overlay",
            )
            self.update_selection(
                df,
                skewer_params,
                col_names,
                psets_vertical_ordering_df,
                curr_selection,
            )

        def get_cds_dict(
            self,
            df,
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
        ):
            df_cb_combined = None
            for col_name in col_names:
                df_cb = (
                    df.groupby(col_name)
                    .size()
                    .to_frame("width")
                    .reset_index(level=0)
                    .rename(columns={col_name: "cluster_id"})
                )
                df_cb["width"] *= skewer_params["width_per_count"]
                df_cb["col_name"] = col_name
                df_cb["left"] = (
                    col_names.get_col_id(col_name) - skewer_params["bar_width"]
                )
                df_cb["right"] = (
                    col_names.get_col_id(col_name) + skewer_params["bar_width"]
                )
                df_cb["line_color"] = df_cb.apply(
                    lambda row: self.get_line_color(
                        df,
                        skewer_params,
                        psets_vertical_ordering_df,
                        curr_selection,
                        col_name,
                        row["cluster_id"],
                    ),
                    axis=1,
                )
                df_cb["bottom"] = df_cb.apply(
                    lambda row: psets_vertical_ordering_df[
                        psets_vertical_ordering_df["label"]
                        == (col_name, row["cluster_id"])
                    ]["y_start"].values[0],
                    axis=1,
                )
                df_cb["top"] = df_cb.apply(
                    lambda row: psets_vertical_ordering_df[
                        psets_vertical_ordering_df["label"]
                        == (col_name, row["cluster_id"])
                    ]["y_end"].values[0],
                    axis=1,
                )
                if curr_selection["color_col_name"] == None:
                    df_cb["fill_color"] = df_cb["line_color"]
                    df_cb["fill_alpha"] = 0.5
                else:
                    df_cb["fill_color"] = "gray"
                    df_cb["fill_alpha"] = 0.2
                # print(df_cb)
                # if len(df_cb.index) < 2:
                #     df_cb["bottom"] = 0
                # else:
                #     df_cb["bottom"] = (
                #         df_cb.index
                #         * skewer_params["spacing_ratio"]
                #         / (len(df_cb.index) - 1)
                #     )
                # df_cb["top"] = df_cb["bottom"] + df_cb["width"].cumsum()
                # df_cb["bottom"] = df_cb["top"] - df_cb["width"]
                if not isinstance(df_cb_combined, pd.DataFrame):
                    df_cb_combined = df_cb
                else:
                    df_cb_combined = pd.concat(
                        [df_cb_combined, df_cb], ignore_index=True
                    )
            return df_cb_combined[self.glyph_vars].to_dict("list")

        def update_selection(
            self,
            df,
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
            old_selection=None,
        ):
            self.glyph.data_source.data = self.get_cds_dict(
                df,
                skewer_params,
                col_names,
                psets_vertical_ordering_df,
                curr_selection,
            )

        def get_line_color(
            self,
            df,
            skewer_params,
            psets_vertical_ordering_df,
            curr_selection,
            col_name,
            cluster_id,
        ):
            if (
                col_name != curr_selection["color_col_name"]
                and curr_selection["color_col_name"] != None
            ):
                return skewer_params["cluster_bars_default_line_color"]
            if (
                len(curr_selection["cluster_ids"]) == 0
                or cluster_id in curr_selection["cluster_ids"]
            ) and curr_selection["color_col_name"] != None:
                return (df[df[col_name] == cluster_id].iloc[0])[
                    skewer_params["color_col_name"]
                ]
            if curr_selection["color_col_name"] == None:
                return psets_vertical_ordering_df[
                    psets_vertical_ordering_df["label"] == (col_name, cluster_id)
                ]["color"].values[0]
            return skewer_params["cluster_bars_filtered_out_line_color"]

    class alluvial_edges:
        def __init__(
            self,
            p,
            df,
            df_filtered,
            psets_vertical_ordering_df,
            psets_color,
            skewer_params,
            col_names,
            curr_selection,
        ):
            self.resolution = 100
            self.glyph_vars = ["xs", "ys", "fill_color", "alpha"]
            self.rbg_edge_alpha_highlight_active = 1
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.patches(
                source=src,
                xs="xs",
                ys="ys",
                fill_color="fill_color",
                line_color=None,
                line_width=0,
                alpha="alpha",
            )
            self.update_selection(
                df,
                df_filtered,
                psets_vertical_ordering_df,
                psets_color,
                skewer_params,
                col_names,
                curr_selection,
            )

        def get_cds_dict(
            self,
            df_non_filtered,
            df,
            col_names,
            color_col_name,
            width_per_count,
            psets_vertical_ordering_df,
            psets_color,
            bar_width,
            curr_selection,
        ):
            if len(col_names) < 2:
                # No alluvial edges if number of columns less than 2
                return to_empty_dict(self.glyph_vars)

            df_edges_combined = None
            for col_name_0, col_name_1 in col_names.get_neighbouring_pairs_l2r():
                col_id_0 = col_names.get_col_id(col_name_0)
                col_id_1 = col_names.get_col_id(col_name_1)

                df_se = (
                    df.groupby([color_col_name, col_name_0, col_name_1])
                    .size()
                    .to_frame("width")
                )

                ###############################################################################################
                # Reducing Edge Crossings during runtime by rearranging the Alluvial Edges.
                # (sort by
                # 1. color - based on the order of color from bottom to top in the selected partition,
                # 2. Location of start point - order of set on the left
                # 3. Location of end point - order of set to the right)
                # color_sort_order = color_palette(
                #     df_non_filtered[color_col_name].unique().size
                # )
                if curr_selection["color_col_name"] != None:
                    color_sort_order = get_color_sort_order(
                        psets_color, curr_selection["color_col_name"]
                    )
                    df_se = df_se.reset_index()
                    df_se["color_index"] = df_se.apply(
                        lambda row: color_sort_order.index(row[color_col_name]), axis=1
                    )
                else:
                    df_se = df_se.reset_index()
                    df_se["color_index"] = 0

                df_se[col_name_0 + "_vertical_order_ref_val"] = df_se.apply(
                    lambda row: psets_vertical_ordering_df[
                        psets_vertical_ordering_df["label"]
                        == (col_name_0, row[col_name_0])
                    ]["y_start"].values[0],
                    axis=1,
                )
                df_se[col_name_1 + "_vertical_order_ref_val"] = df_se.apply(
                    lambda row: psets_vertical_ordering_df[
                        psets_vertical_ordering_df["label"]
                        == (col_name_1, row[col_name_1])
                    ]["y_start"].values[0],
                    axis=1,
                )

                df_se = df_se.sort_values(
                    by=[
                        "color_index",
                        col_name_0 + "_vertical_order_ref_val",
                        col_name_1 + "_vertical_order_ref_val",
                    ]
                )
                df_se.set_index([color_col_name, col_name_0, col_name_1], inplace=True)
                ###############################################################################################

                if self.rbg_edge_alpha_highlight_active in [0, 2]:
                    df_FMI = calc_FMI(df, col_name_0, col_name_1)
                    df_se = df_se.join(df_FMI, on=[col_name_0, col_name_1])
                    df_se["alpha"] = (
                        (1 - df_se["FMI"])
                        if self.rbg_edge_alpha_highlight_active == 0
                        else df_se["FMI"]
                    )
                else:
                    df_se["alpha"] = 0.5

                df_se["width"] *= width_per_count
                df_se["x0"] = col_id_0 - (bar_width if col_id_0 == 0 else 0)
                df_se["x1"] = col_id_1 + (
                    bar_width if col_id_1 == (len(col_names) - 1) else 0
                )
                df_se["y0_start"] = None
                df_se["y1_start"] = None
                df_se["fill_color"] = df_se.index.get_level_values(0)

                # y_start_0 = copy.deepcopy(y_start[col_id_0])
                # y_start_1 = copy.deepcopy(y_start[col_id_1])

                # Filter the DataFrame based on the condition
                filtered_setwise_position_df_0 = psets_vertical_ordering_df[
                    psets_vertical_ordering_df["partition_col_name"] == col_name_0
                ]
                # Create a dictionary using "partition_set_categorical_value" as keys and "y_start" as values
                filtered_setwise_position_dict_0 = dict(
                    zip(
                        filtered_setwise_position_df_0[
                            "partition_set_categorical_value"
                        ],
                        filtered_setwise_position_df_0["y_start"],
                    )
                )
                # Filter the DataFrame based on the condition
                filtered_setwise_position_df_1 = psets_vertical_ordering_df[
                    psets_vertical_ordering_df["partition_col_name"] == col_name_1
                ]
                # Create a dictionary using "partition_set_categorical_value" as keys and "y_start" as values
                filtered_setwise_position_dict_1 = dict(
                    zip(
                        filtered_setwise_position_df_1[
                            "partition_set_categorical_value"
                        ],
                        filtered_setwise_position_df_1["y_start"],
                    )
                )

                y_start_0 = copy.deepcopy(filtered_setwise_position_dict_0)
                y_start_1 = copy.deepcopy(filtered_setwise_position_dict_1)
                # print(y_start[col_id_0])
                # print(
                #     list(
                #         psets_vertical_ordering_df[
                #             psets_vertical_ordering_df["partition_col_name"] == col_name_0
                #         ]["y_start"].values
                #     )
                # )

                # Define a custom function to update the DataFrame
                def update_y_start(row):
                    nonlocal y_start_0, y_start_1
                    y0 = y_start_0[row.name[1]]
                    y1 = y_start_1[row.name[2]]
                    row["y0_start"] = y0
                    row["y1_start"] = y1
                    y_start_0[row.name[1]] += row["width"]
                    y_start_1[row.name[2]] += row["width"]
                    return row

                # Apply the custom function to update the DataFrame
                df_se = df_se.apply(update_y_start, axis=1)

                df_se = df_se.reset_index(drop=True)
                if not isinstance(df_edges_combined, pd.DataFrame):
                    df_edges_combined = df_se
                else:
                    df_edges_combined = pd.concat(
                        [df_edges_combined, df_se], ignore_index=True
                    )
            df_edges_combined["xs"] = df_edges_combined.apply(
                lambda row: self.get_edges_dict_calc_xs(row["x0"], row["x1"]), axis=1
            )
            df_edges_combined["ys"] = df_edges_combined.apply(
                lambda row: self.get_edges_dict_calc_ys(
                    row["x0"], row["x1"], row["y0_start"], row["y1_start"], row["width"]
                ),
                axis=1,
            )
            return df_edges_combined[self.glyph_vars].to_dict("list")

        def update_selection(
            self,
            df,
            df_filtered,
            psets_vertical_ordering_df,
            psets_color,
            skewer_params,
            col_names,
            curr_selection,
            old_selection=None,
        ):
            self.glyph.data_source.data = self.get_cds_dict(
                df,
                df_filtered,
                col_names,
                skewer_params["color_col_name"],
                skewer_params["width_per_count"],
                psets_vertical_ordering_df,
                psets_color,
                skewer_params["bar_width"],
                curr_selection,
            )

        def get_edges_dict_calc_xs(self, x0, x1):
            x_smooth = np.linspace(x0, x1, self.resolution)
            return np.concatenate([x_smooth, x_smooth[::-1]])

        def get_edges_dict_calc_ys(self, x0, x1, y0_start, y1_start, width):
            x_non_smooth = np.concatenate(
                [
                    np.linspace(x0, x0 + (x1 - x0) * 0.1, 10),
                    np.linspace(x0 + (x1 - x0) * 0.9, x1, 10),
                ]
            )
            y_non_smooth = np.concatenate([[y0_start] * 10, [y1_start] * 10])
            spl = InterpolatedUnivariateSpline(x_non_smooth, y_non_smooth)
            x_smooth = np.linspace(x0, x1, self.resolution)
            y1_smooth = spl(x_smooth)
            y2_smooth = y1_smooth + width
            return np.concatenate([y1_smooth, y2_smooth[::-1]])

    class radio_buttons:
        def __init__(self, p, skewer_params, col_names, curr_selection):
            self.glyphs = {}
            self.glyphs["colorful_boundary"] = self.rb_colorful_boundary(
                p,
                skewer_params,
                col_names,
                curr_selection,
            )
            self.glyphs["elliptical_buttons"] = self.rb_elliptical_buttons(
                p,
                skewer_params,
                col_names,
                curr_selection,
            )
            self.glyphs["button_labels"] = self.rb_button_labels(
                p,
                skewer_params,
                col_names,
                curr_selection,
            )

        def update_selection(
            self, skewer_params, col_names, curr_selection, old_selection
        ):
            for glyph_type in self.glyphs.keys():
                self.glyphs[glyph_type].update_selection(
                    skewer_params, col_names, curr_selection, old_selection
                )

        class rb_elliptical_buttons:
            def __init__(self, p, skewer_params, col_names, curr_selection):
                self.glyph_vars = ["x", "fill_color", "hatch_pattern"]
                src = ColumnDataSource(to_empty_dict(self.glyph_vars))
                self.glyph = p.ellipse(
                    source=src,
                    x="x",
                    y=skewer_params["rb_ellipse_y"],
                    width=skewer_params["rb_ellipse_width"],
                    height=skewer_params["rb_ellipse_height"],
                    fill_color="fill_color",
                    line_color=skewer_params["rb_line_color"],
                    line_width=skewer_params["rb_line_width"],
                    hatch_color=skewer_params["rb_line_color"],
                    hatch_pattern="hatch_pattern",
                    hatch_alpha=0.5,
                    hatch_scale=5,
                )
                self.update_selection(skewer_params, col_names, curr_selection)

            def get_cds_dict(self, skewer_params, col_names, curr_selection):
                df = pd.DataFrame(col_names, columns=["col_names"])
                df["x"] = df.apply(
                    lambda row: col_names.get_col_id(row["col_names"]),
                    axis=1,
                )
                df["fill_color"] = df.apply(
                    lambda row: skewer_params["rb_fill_color_selected"]
                    if row["col_names"] == curr_selection["color_col_name"]
                    else skewer_params["rb_fill_color_unselected"],
                    axis=1,
                )
                df["hatch_pattern"] = df.apply(
                    lambda row: skewer_params["rb_hatch_pattern_filtered_column"]
                    if (
                        row["col_names"] == curr_selection["color_col_name"]
                        and len(curr_selection["cluster_ids"]) != 0
                    )
                    else skewer_params["rb_hatch_pattern_not_filtered_column"],
                    axis=1,
                )
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, skewer_params, col_names, curr_selection, old_selection=None
            ):
                self.glyph.data_source.data = self.get_cds_dict(
                    skewer_params, col_names, curr_selection
                )

        class rb_button_labels:
            def __init__(self, p, skewer_params, col_names, curr_selection):
                self.glyph_vars = ["x", "text", "text_font_size", "text_color"]
                src = ColumnDataSource(to_empty_dict(self.glyph_vars))
                self.glyph = LabelSet(
                    source=src,
                    x="x",
                    y=skewer_params["rb_labels_y"],
                    text="text",
                    text_font_size="text_font_size",
                    text_align="center",
                    text_font_style="bold",
                    text_color="text_color",
                )
                p.add_layout(self.glyph)
                self.update_selection(skewer_params, col_names, curr_selection)

            def get_cds_dict(self, skewer_params, col_names, curr_selection):
                df = pd.DataFrame(col_names, columns=["col_names"])
                df["x"] = df.apply(
                    lambda row: col_names.get_col_id(row["col_names"]),
                    axis=1,
                )
                # TODO: Handle larger strings and numbers
                # for eg. ("{:.1E}".format(val) if len(str(val)) >= 5 else str(val))
                df["text"] = df["col_names"]
                df["text_font_size"] = df.apply(
                    lambda row: "7pt" if len(row["text"]) > 3 else "10pt",
                    axis=1,
                )
                df["text_color"] = df.apply(
                    lambda row: skewer_params["rb_fill_color_unselected"]
                    if row["col_names"] == curr_selection["color_col_name"]
                    else skewer_params["rb_fill_color_selected"],
                    axis=1,
                )
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, skewer_params, col_names, curr_selection, old_selection=None
            ):
                self.glyph.source.data = self.get_cds_dict(
                    skewer_params, col_names, curr_selection
                )

        class rb_colorful_boundary:
            def __init__(self, p, skewer_params, col_names, curr_selection):
                self.glyph_vars = ["x0", "x1", "y0", "y1", "line_color"]
                self.n_colors = 256
                src = ColumnDataSource(to_empty_dict(self.glyph_vars))
                self.glyph = p.segment(
                    source=src,
                    x0="x0",
                    x1="x1",
                    y0="y0",
                    y1="y1",
                    line_color="line_color",
                    line_width=3,
                )
                self.update_selection(skewer_params, col_names, curr_selection)

            def get_cds_dict(self, skewer_params, center_x):
                if center_x == None:
                    return to_empty_dict(self.glyph_vars)
                center_y = skewer_params["rb_ellipse_y"]
                angle_linspace = np.linspace(
                    0 + np.pi / 2, 2 * np.pi + np.pi / 2, num=self.n_colors + 1
                )
                df = pd.DataFrame(
                    list(zip(angle_linspace[:-1], angle_linspace[1:])),
                    columns=["angle0", "angle1"],
                )
                df["x0"] = df.apply(
                    lambda row: skewer_params["rb_ellipse_bondary_halfwidth"]
                    * np.cos(row["angle0"])
                    + center_x,
                    axis=1,
                )
                df["x1"] = df.apply(
                    lambda row: skewer_params["rb_ellipse_bondary_halfwidth"]
                    * np.cos(row["angle1"])
                    + center_x,
                    axis=1,
                )
                df["y0"] = df.apply(
                    lambda row: skewer_params["rb_ellipse_bondary_halfheight"]
                    * np.sin(row["angle0"])
                    + center_y,
                    axis=1,
                )
                df["y1"] = df.apply(
                    lambda row: skewer_params["rb_ellipse_bondary_halfheight"]
                    * np.sin(row["angle1"])
                    + center_y,
                    axis=1,
                )
                df["line_color"] = color_palette(self.n_colors, shuffle=False)
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, skewer_params, col_names, curr_selection, old_selection=None
            ):
                center_x = (
                    col_names.get_col_id(curr_selection["color_col_name"])
                    if curr_selection["color_col_name"] != None
                    else None
                )
                self.glyph.data_source.data = self.get_cds_dict(skewer_params, center_x)

    def update_selection(
        self,
        df,
        df_filtered,
        psets_vertical_ordering_df,
        psets_color,
        skewer_params,
        col_names,
        curr_selection,
        old_selection,
    ):
        timer_obj = timer("Updating Alluvial Diagram")
        self.rb_obj.update_selection(
            skewer_params, col_names, curr_selection, old_selection
        )
        self.alluvial_edges_obj.update_selection(
            df,
            df_filtered,
            psets_vertical_ordering_df,
            psets_color,
            skewer_params,
            col_names,
            curr_selection,
            old_selection,
        )
        self.alluvial_cluster_bars_obj.update_selection(
            df,
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
            old_selection,
        )
        self.number_line_selection_ellipse_obj.update_selection(
            self.number_line_df,
            df,
            df_filtered,
            skewer_params,
            col_names,
            curr_selection,
            old_selection,
        )
        timer_obj.done()

    def generate_figure(self, skewer_params):
        p = figure(
            width=skewer_params["widthspx_alluvial"],
            height=skewer_params["heightspx_alluvial"],
            tools="",
            y_range=(
                skewer_params["alluvial_y_start"],
                skewer_params["alluvial_y_end"],
            ),
        )
        p.toolbar.logo = None
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.xaxis.visible = False
        # p.yaxis.visible = False
        p.yaxis.axis_line_color = None
        p.outline_line_color = None
        p.min_border = 0
        p.x_range.range_padding = 0.03
        p.yaxis.major_tick_line_color = "lightgray"

        return p


class cim:
    def __init__(
        self,
        x_range,
        df,
        skewer_params,
        psets_color,
        col_names,
        curr_selection,
    ):
        self.p_normal, self.p_inverted = self.generate_figures(x_range, skewer_params)
        draw_vlines(self.p_normal, col_names)
        draw_vlines(self.p_inverted, col_names)
        self.draw_axis_line()
        # self.cim_setwise_details_or_not_cbgrp = CheckboxButtonGroup(
        #     labels=["Setwise", "Setwise"],
        #     active=[0, 1],
        #     orientation="vertical",
        #     inline=True,
        # )
        self.p_normal.yaxis.axis_label = "Overlap Measure"
        self.overlap_measure = self.cim_overlap(
            self.p_normal,
            x_range,
            df,
            skewer_params,
            psets_color,
            col_names,
            curr_selection,
        )
        self.p_inverted.yaxis.axis_label = "Split Measure"
        self.split_measure = self.cim_split(
            self.p_inverted,
            df,
            skewer_params,
            col_names,
            curr_selection,
        )

    class cim_overlap:
        def __init__(
            self,
            p,
            x_range,
            df,
            skewer_params,
            psets_color,
            col_names,
            curr_selection,
        ):
            self.background_glyph_vars = ["top", "bottom", "fill_color"]
            background_src = ColumnDataSource(to_empty_dict(self.background_glyph_vars))
            self.background_glyph = p.quad(
                source=background_src,
                top="top",
                bottom="bottom",
                left=-0.5,
                right=len(col_names) - 0.5,
                fill_color="fill_color",
                fill_alpha=0.15,
                line_width=0,
                # line_color=None,
            )
            self.glyph_vars = ["left", "right", "top", "bottom", "fill_color"]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.quad(
                source=src,
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                line_width=1,
                line_alpha=0.25,
                line_color="white",
                fill_color="fill_color",
                fill_alpha=0.5,
            )
            self.update_selection(
                df,
                df,
                skewer_params,
                psets_color,
                col_names,
                curr_selection,
            )

        def get_cds_dict(
            self,
            df_non_filtered,
            df,
            skewer_params,
            psets_color,
            col_names,
            wrt_col_name,
            bool_measure_only=False,
        ):
            if not bool_measure_only:
                if wrt_col_name == None:
                    return to_empty_dict(self.glyph_vars), to_empty_dict(
                        self.background_glyph_vars
                    )
                df_cim_overlap_combined = None
            else:
                if wrt_col_name == None:
                    return 0
                dissimilarity_dict = {}
            df_total_count = len(df.index)
            for col_name in col_names:
                if col_name == wrt_col_name:
                    if bool_measure_only:
                        dissimilarity_dict[(wrt_col_name, col_name)] = 0
                    continue
                df_cim_overlap = (
                    df[[col_name, wrt_col_name, skewer_params["color_col_name"]]]
                    .groupby([col_name, wrt_col_name])
                    .agg(
                        count=(col_name, "size"),
                        fill_color=(skewer_params["color_col_name"], "first"),
                    )
                )
                df_cim_overlap_temp = (
                    df_cim_overlap[["count"]]
                    .groupby(level=0)
                    .sum()
                    .rename(columns={"count": "count_total"})
                )
                df_cim_overlap = df_cim_overlap.join(
                    df_cim_overlap_temp.reindex(df_cim_overlap.index, level=0)
                )
                df_cim_overlap["cim_overlap_individual"] = (
                    (df_cim_overlap["count_total"] - df_cim_overlap["count"])
                    * df_cim_overlap["count"]
                ) / df_cim_overlap["count_total"]

                df_cim_overlap["cim_overlap"] = (
                    df_cim_overlap["cim_overlap_individual"] / df_total_count
                )

                if bool_measure_only:
                    dissimilarity_dict[(wrt_col_name, col_name)] = df_cim_overlap[
                        "cim_overlap"
                    ].sum()
                    continue

                df_cim_overlap = (
                    df_cim_overlap.reset_index()
                    .sort_values([wrt_col_name], ascending=[True])
                    .set_index([wrt_col_name, col_name])
                )

                df_cim_overlap_temp = (
                    df_cim_overlap[
                        ["cim_overlap", "cim_overlap_individual", "count", "fill_color"]
                    ]
                    .groupby(level=0)
                    .agg(
                        cim_overlap=("cim_overlap", "sum"),
                        cim_overlap_individual=("cim_overlap_individual", "sum"),
                        count=("count", "sum"),
                        fill_color=("fill_color", "first"),
                    )
                )
                df_cim_overlap_temp["cim_overlap_individual"] = (
                    df_cim_overlap_temp["cim_overlap_individual"]
                    / df_cim_overlap_temp["count"]
                )

                # df_cim_overlap_temp["width_percentage"] = (
                #     df_cim_overlap_temp["cim_overlap"] / df_cim_overlap_temp["count"]
                # )
                # df_cim_overlap_temp["width_percentage"] = (
                #     df_cim_overlap_temp["width_percentage"]
                #     / df_cim_overlap_temp["width_percentage"].max()
                # )
                df_cim_overlap_temp["width_percentage"] = 1
                df_cim_overlap_temp["bar_halfwidth"] = df_cim_overlap_temp[
                    "width_percentage"
                ] * (skewer_params["cim_bar_width"] / 2)
                df_cim_overlap_temp["left"] = (
                    col_names.get_col_id(col_name)
                    - df_cim_overlap_temp["bar_halfwidth"]
                )
                df_cim_overlap_temp["right"] = (
                    col_names.get_col_id(col_name)
                    + df_cim_overlap_temp["bar_halfwidth"]
                )
                # df_cim_overlap_temp = df_cim_overlap_temp.sort_values(
                #     ["width_percentage"], ascending=[True]
                # )

                ###############################################################################################
                # (sort by color - based on the order of color from bottom to top in the selected partition)
                color_sort_order = get_color_sort_order(psets_color, wrt_col_name)

                df_cim_overlap_temp = df_cim_overlap_temp.reset_index()
                df_cim_overlap_temp["color_index"] = df_cim_overlap_temp.apply(
                    lambda row: color_sort_order.index(row["fill_color"]),
                    axis=1,
                )
                df_cim_overlap_temp = df_cim_overlap_temp.sort_values(
                    by=["color_index"]
                )
                df_cim_overlap_temp.set_index([wrt_col_name], inplace=True)
                ###############################################################################################

                df_cim_overlap_temp["count_cumsum"] = df_cim_overlap_temp[
                    "count"
                ].cumsum()
                # df_cim_overlap_temp["top"] = df_cim_overlap_temp["cim_overlap"].cumsum()
                # df_cim_overlap_temp["bottom"] = (
                #     df_cim_overlap_temp["top"] - df_cim_overlap_temp["cim_overlap"]
                # )

                df_cim_overlap_temp["bottom"] = (
                    df_cim_overlap_temp["count_cumsum"] - df_cim_overlap_temp["count"]
                ) / df_total_count
                df_cim_overlap_temp["top"] = (
                    df_cim_overlap_temp["count_cumsum"] / df_total_count
                )
                df_cim_overlap_temp["top"] = (
                    df_cim_overlap_temp["top"] - df_cim_overlap_temp["bottom"]
                ) * df_cim_overlap_temp["cim_overlap_individual"] + df_cim_overlap_temp[
                    "bottom"
                ]

                # print(df_cim_overlap_temp)

                # df_cim_overlap_temp["top"] = df_cim_overlap_temp["cim_overlap"].cumsum()
                # df_cim_overlap_temp["bottom"] = (
                #     df_cim_overlap_temp["top"] - df_cim_overlap_temp["cim_overlap"]
                # )

                df_cim_overlap = df_cim_overlap_temp

                if not isinstance(df_cim_overlap_combined, pd.DataFrame):
                    df_cim_overlap_combined = df_cim_overlap
                    df_cim_overlap_background = copy.deepcopy(
                        df_cim_overlap[["fill_color", "bottom"]]
                    )
                    background_top_list = list(df_cim_overlap["bottom"])[1:]
                    background_top_list.append(1)
                    df_cim_overlap_background["top"] = background_top_list
                else:
                    df_cim_overlap_combined = pd.concat(
                        [df_cim_overlap_combined, df_cim_overlap], ignore_index=True
                    )
            if bool_measure_only:
                return dissimilarity_dict
            return df_cim_overlap_combined[self.glyph_vars].to_dict(
                "list"
            ), df_cim_overlap_background[self.background_glyph_vars].to_dict("list")

        def update_selection(
            self,
            df_non_filtered,
            df,
            skewer_params,
            psets_color,
            col_names,
            curr_selection,
            old_selection=None,
        ):
            timer_obj = timer("Updating CIM Overlap")
            (
                self.glyph.data_source.data,
                self.background_glyph.data_source.data,
            ) = self.get_cds_dict(
                df_non_filtered,
                df,
                skewer_params,
                psets_color,
                col_names,
                curr_selection["color_col_name"],
            )
            timer_obj.done()

        def calc_dissimilarity_mat(self, df, skewer_params, col_names):
            dissimilarity_dict = {}
            for col_name in col_names:
                dissimilarity_dict = dissimilarity_dict | self.get_cds_dict(
                    df, skewer_params, col_names, col_name, bool_measure_only=True
                )
            for col_name_0 in col_names:
                for col_name_1 in col_names:
                    if (
                        dissimilarity_dict[(col_name_0, col_name_1)]
                        != dissimilarity_dict[(col_name_1, col_name_0)]
                    ):
                        average_cim = (
                            dissimilarity_dict[(col_name_0, col_name_1)]
                            + dissimilarity_dict[(col_name_0, col_name_1)]
                        ) / 2
                        dissimilarity_dict[(col_name_0, col_name_1)] = average_cim
                        dissimilarity_dict[(col_name_1, col_name_0)] = average_cim
            dissimilarity_np = np.empty(shape=(len(col_names), len(col_names)))
            for i, col_name_0 in enumerate(col_names):
                for j, col_name_1 in enumerate(col_names):
                    dissimilarity_np[i][j] = dissimilarity_dict[
                        (col_name_0, col_name_1)
                    ]
            return dissimilarity_np

    class cim_split:
        def __init__(self, p, df, skewer_params, col_names, curr_selection):
            self.glyph_vars = ["left", "right", "top", "bottom", "fill_color"]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.quad(
                source=src,
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                line_width=1,
                line_alpha=0.25,
                line_color="white",
                fill_color="fill_color",
                fill_alpha=0.75,
            )
            self.update_selection(
                df,
                skewer_params,
                col_names,
                curr_selection,
            )

        def get_cds_dict(self, df, skewer_params, col_names, wrt_col_name):
            if wrt_col_name == None:
                return to_empty_dict(self.glyph_vars)
            df_cim_split_combined = None
            df_total_count = len(df.index)
            for col_name in col_names:
                if col_name == wrt_col_name:
                    continue
                df_cim_split = (
                    df[[wrt_col_name, col_name, skewer_params["color_col_name"]]]
                    .groupby([wrt_col_name, col_name])
                    .agg(
                        count=(col_name, "size"),
                        fill_color=(skewer_params["color_col_name"], "first"),
                    )
                    .reset_index()
                    .sort_values([wrt_col_name, "count"], ascending=[True, False])
                    .set_index([wrt_col_name, col_name])
                )
                df_cim_split["count_cumsum"] = (
                    df_cim_split[["count"]].groupby(level=0).cumsum()["count"]
                )
                df_cim_split_temp = (
                    df_cim_split[["count"]]
                    .groupby(level=0)
                    .sum()
                    .rename(columns={"count": "count_total"})
                )
                df_cim_split = df_cim_split.join(
                    df_cim_split_temp.reindex(df_cim_split.index, level=0)
                )
                df_cim_split["count_remaining"] = (
                    df_cim_split["count_total"]
                    - df_cim_split["count_cumsum"]
                    + df_cim_split["count"]
                )
                df_cim_split["cim_split_measure"] = (
                    (df_cim_split["count_remaining"] - df_cim_split["count"])
                    * df_cim_split["count"]
                    / (df_cim_split["count_remaining"] * df_total_count)
                )
                df_cim_split_temp = (
                    df_cim_split[["cim_split_measure", "count", "fill_color"]]
                    .groupby(level=0)
                    .agg(
                        cim_split_measure=("cim_split_measure", "sum"),
                        count=("count", "sum"),
                        fill_color=("fill_color", "first"),
                    )
                )
                df_cim_split_temp["width_percentage"] = (
                    df_cim_split_temp["cim_split_measure"] / df_cim_split_temp["count"]
                )
                df_cim_split_temp["width_percentage"] = (
                    df_cim_split_temp["width_percentage"]
                    / df_cim_split_temp["width_percentage"].max()
                )
                df_cim_split_temp["bar_halfwidth"] = df_cim_split_temp[
                    "width_percentage"
                ] * (skewer_params["cim_bar_width"] / 2)
                df_cim_split_temp["left"] = (
                    col_names.get_col_id(col_name) - df_cim_split_temp["bar_halfwidth"]
                )
                df_cim_split_temp["right"] = (
                    col_names.get_col_id(col_name) + df_cim_split_temp["bar_halfwidth"]
                )
                df_cim_split_temp = df_cim_split_temp.sort_values(
                    ["width_percentage"], ascending=[True]
                )
                df_cim_split_temp["top"] = df_cim_split_temp[
                    "cim_split_measure"
                ].cumsum()
                df_cim_split_temp["bottom"] = (
                    df_cim_split_temp["top"] - df_cim_split_temp["cim_split_measure"]
                )
                # df_cim_split_temp["fill_color"] = "gray"
                # df_cim_split_temp["fill_color"] = df_cim_split_temp.apply(
                #     lambda row: (df[df[wrt_col_name] == row.name].iloc[0])[
                #         skewer_params["color_col_name"]
                #     ],
                #     axis=1,
                # )
                df_cim_split = df_cim_split_temp
                if not isinstance(df_cim_split_combined, pd.DataFrame):
                    df_cim_split_combined = df_cim_split
                else:
                    df_cim_split_combined = pd.concat(
                        [df_cim_split_combined, df_cim_split], ignore_index=True
                    )
            return df_cim_split_combined[self.glyph_vars].to_dict("list")

        def update_selection(
            self, df, skewer_params, col_names, curr_selection, old_selection=None
        ):
            timer_obj = timer("Updating CIM Split")
            self.glyph.data_source.data = self.get_cds_dict(
                df, skewer_params, col_names, curr_selection["color_col_name"]
            )
            timer_obj.done()

    def update_selection(
        self,
        df_non_filtered,
        df_filtered,
        skewer_params,
        psets_color,
        col_names,
        curr_selection,
        old_selection,
    ):
        self.overlap_measure.update_selection(
            df_non_filtered,
            df_filtered,
            skewer_params,
            psets_color,
            col_names,
            curr_selection,
            old_selection,
        )
        self.split_measure.update_selection(
            df_filtered, skewer_params, col_names, curr_selection, old_selection
        )

    def generate_figure(self, x_range, skewer_params):
        if x_range != None:
            p = figure(
                width=skewer_params["widthspx_cim"],
                height=skewer_params["heightspx_cim"],
                x_range=x_range,
                tools="",
            )
        else:
            p = figure(
                width=skewer_params["widthspx_cim"],
                height=skewer_params["heightspx_cim"],
                tools="",
            )
        p.toolbar.logo = None
        p.xgrid.visible = False
        p.xaxis.visible = False
        # p.xaxis.ticker = np.array(range(len(col_names)))
        p.outline_line_color = None
        p.min_border = 0
        p.xaxis.major_label_text_font_size = "0pt"
        p.x_range.range_padding = 0
        # p.xgrid.grid_line_color = None
        p.ygrid.visible = False
        p.yaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        return p

    def generate_figures(self, x_range, skewer_params):
        p_normal = self.generate_figure(x_range, skewer_params)
        p_inverted = self.generate_figure(x_range, skewer_params)
        p_normal.y_range.start = -0.001
        p_normal.y_range.end = 1.01
        p_inverted.y_range.start = 1.01
        p_inverted.y_range.end = -0.001
        p_inverted.y_range.flipped = True
        return p_normal, p_inverted

    def draw_axis_line(self):
        axis_line = Span(
            location=0,
            dimension="width",
            line_color="black",
            line_width=1,
            level="overlay",
        )
        self.p_normal.renderers.extend([axis_line])
        self.p_inverted.renderers.extend([axis_line])


# class metamap_edit_dist_pt_grps:
#     def __init__(self, df, skewer_params, col_names, curr_selection):
#         self.p1 = self.generate_figure(
#             skewer_params,
#             title="Cluster Split Analysis - Left: Stable Clusters; Right: Cluster splits",
#         )
#         self.p2 = self.generate_figure(
#             skewer_params,
#             title="Clusters Pair Overlap Analysis - Left: overlapping; Right: non-overlapping",
#         )
#         self.glyph_vars = ["left", "right", "top", "bottom", "fill_color"]
#         src1 = ColumnDataSource(to_empty_dict(self.glyph_vars))
#         src2 = ColumnDataSource(to_empty_dict(self.glyph_vars))
#         self.glyph1 = self.p1.quad(
#             source=src1,
#             left="left",
#             right="right",
#             top="top",
#             bottom="bottom",
#             fill_color="fill_color",
#             line_color=None,
#             alpha=0.6,
#         )
#         self.glyph2 = self.p2.quad(
#             source=src2,
#             left="left",
#             right="right",
#             top="top",
#             bottom="bottom",
#             fill_color="fill_color",
#             line_color=None,
#             alpha=0.6,
#         )
#         self.update_selection(df, skewer_params, col_names, curr_selection)

#     def generate_figure(self, skewer_params, title):
#         p = figure(
#             width=skewer_params["widthspx_ndimplot"],
#             height=skewer_params["heightspx_editdist"],
#             tools=[
#                 BoxZoomTool(),
#                 PanTool(),
#                 WheelZoomTool(),
#                 ResetTool(),
#             ],
#             title=title,
#         )
#         p.toolbar.logo = None
#         p.min_border = 0
#         p.xaxis.axis_label = "Edit distance"
#         p.yaxis.axis_label = "Data point pair count"
#         return p

#     def get_cds_dict(self, df, skewer_params, col_names, curr_selection):
#         df_edit_dist = df[list(col_names) + [skewer_params["color_col_name"]]].copy()
#         df_edit_dist["edit_string"] = ""
#         for col_name in col_names:
#             list_unique_vals = get_unique_vals(df_edit_dist, col_name)
#             df_edit_dist["edit_string"] = df_edit_dist.apply(
#                 lambda row: row["edit_string"]
#                 + chr(list_unique_vals.index(row[col_name])),
#                 axis=1,
#             )
#         df_edit_dist_temp = (
#             df_edit_dist.groupby(["edit_string"])
#             .agg(
#                 count=(col_name, "size"),
#                 color_col_name=(curr_selection["color_col_name"], "first"),
#                 fill_color=(skewer_params["color_col_name"], "first"),
#             )
#             .reset_index()
#             .rename(
#                 columns={"color_col_name": curr_selection["color_col_name"]},
#                 inplace=False,
#             )
#         )
#         df_edit_dist_temp["index_col"] = df_edit_dist_temp.index
#         df_edit_dist_temp.rename(
#             columns={curr_selection["color_col_name"]: "cl_id"}, inplace=True
#         )
#         df_edit_dist_temp.loc[:, "key_col"] = 1
#         df_edit_dist_temp = (
#             pd.merge(
#                 df_edit_dist_temp,
#                 df_edit_dist_temp,
#                 on="key_col",
#                 suffixes=("_1", "_2"),
#             )
#             .query("index_col_1 <= index_col_2")
#             .drop("key_col", axis=1)
#             .reset_index(drop=True)
#         )
#         df_edit_dist_temp1 = df_edit_dist_temp.query(
#             "index_col_1 == index_col_2"
#         ).copy()
#         df_edit_dist_temp2 = df_edit_dist_temp.query(
#             "index_col_1 != index_col_2"
#         ).copy()
#         df_edit_dist_temp1["edit_distance"] = 0
#         df_edit_dist_temp1["point_pair_count"] = (
#             df_edit_dist_temp1["count_1"] * (df_edit_dist_temp1["count_1"] - 1) / 2
#         )
#         df_edit_dist_temp2["edit_distance"] = df_edit_dist_temp2.apply(
#             lambda row: compute_edit_distance(
#                 row["edit_string_1"], row["edit_string_2"]
#             ),
#             axis=1,
#         )
#         df_edit_dist_temp2["point_pair_count"] = (
#             df_edit_dist_temp2["count_1"] * df_edit_dist_temp2["count_2"]
#         )
#         df_edit_dist_temp = pd.concat([df_edit_dist_temp1, df_edit_dist_temp2], axis=0)
#         df_edit_dist_temp1 = df_edit_dist_temp.query("cl_id_1 == cl_id_2")[
#             ["cl_id_1", "fill_color_1", "edit_distance", "point_pair_count"]
#         ].copy()
#         df_edit_dist_temp2l = (
#             df_edit_dist_temp.query("cl_id_1 != cl_id_2")[
#                 [
#                     "cl_id_1",
#                     "fill_color_1",
#                     "cl_id_2",
#                     "fill_color_2",
#                     "edit_distance",
#                     "point_pair_count",
#                 ]
#             ]
#             .copy()
#             .groupby(["cl_id_1", "cl_id_2", "edit_distance"])
#             .agg(
#                 n_point_pairs=("point_pair_count", "sum"),
#                 fill_color_1=("fill_color_1", "first"),
#                 fill_color_2=("fill_color_2", "first"),
#             )
#             .reset_index()
#             .sort_values("n_point_pairs", ascending=False)
#         )
#         df_edit_dist_temp2l["top"] = (
#             df_edit_dist_temp2l[["edit_distance", "n_point_pairs"]]
#             .groupby(["edit_distance"])
#             .cumsum()["n_point_pairs"]
#         )
#         df_edit_dist_temp2l["bottom"] = (
#             df_edit_dist_temp2l["top"] - df_edit_dist_temp2l["n_point_pairs"]
#         )
#         df_edit_dist_temp2r = df_edit_dist_temp2l.copy()
#         df_edit_dist_temp2l["left"] = df_edit_dist_temp2l["edit_distance"] - 0.4
#         df_edit_dist_temp2l["right"] = df_edit_dist_temp2l["edit_distance"]
#         df_edit_dist_temp2l["fill_color"] = df_edit_dist_temp2l["fill_color_1"]
#         df_edit_dist_temp2r["left"] = df_edit_dist_temp2r["edit_distance"]
#         df_edit_dist_temp2r["right"] = df_edit_dist_temp2r["edit_distance"] + 0.4
#         df_edit_dist_temp2r["fill_color"] = df_edit_dist_temp2r["fill_color_2"]
#         df_edit_dist_temp2 = pd.concat(
#             [df_edit_dist_temp2l, df_edit_dist_temp2r], axis=0
#         )

#         df_edit_dist_temp1.rename(columns={"fill_color_1": "fill_color"}, inplace=True)
#         df_edit_dist_temp1 = df_edit_dist_temp1.sort_values("cl_id_1")
#         df_edit_dist_temp1 = (
#             df_edit_dist_temp1.groupby(["cl_id_1", "edit_distance"])
#             .agg(
#                 n_point_pairs=("point_pair_count", "sum"),
#                 fill_color=("fill_color", "first"),
#             )
#             .reset_index()
#         )
#         df_edit_dist_temp1["left"] = df_edit_dist_temp1["edit_distance"] - 0.4
#         df_edit_dist_temp1["right"] = df_edit_dist_temp1["edit_distance"] + 0.4
#         df_edit_dist_temp1["top"] = (
#             df_edit_dist_temp1[["edit_distance", "n_point_pairs"]]
#             .groupby(["edit_distance"])
#             .cumsum()["n_point_pairs"]
#         )
#         df_edit_dist_temp1["bottom"] = (
#             df_edit_dist_temp1["top"] - df_edit_dist_temp1["n_point_pairs"]
#         )

#         return df_edit_dist_temp1[self.glyph_vars].to_dict("list"), df_edit_dist_temp2[
#             self.glyph_vars
#         ].to_dict("list")

#     def update_selection(
#         self, df, skewer_params, col_names, curr_selection, old_selection=None
#     ):
#         return  # for faster execution, leave this vis out
#         timer_obj = timer("Updating Meta-Map Edit distance")
#         cds_dict1, cds_dict2 = self.get_cds_dict(
#             df, skewer_params, col_names, curr_selection
#         )
#         self.glyph1.data_source.data = cds_dict1
#         self.glyph2.data_source.data = cds_dict2
#         timer_obj.done()


class ndimplot:
    def __init__(self, df, skewer_params, col_names, curr_selection):
        self.multichoice_cols = MultiChoice(
            title="Select 1 or more columns for the N-dimensional plot",
            value=curr_selection["ndimplot_col_names"],
            options=list(df.columns),
        )
        self.p = self.generate_ndimplot_figure(skewer_params)
        self.glyph = [None] * 4
        for i in range(4):
            self.glyph_vars = self.get_glyph_vars(i)
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            if i == 0:
                self.glyph[i] = self.p.text(
                    source=src,
                    x="x",
                    y="y",
                    text="text",
                    text_color="black",
                    text_align="center",
                    text_baseline="middle",
                )
            elif i in [1, 3]:
                self.glyph[i] = self.p.quad(
                    source=src,
                    left="left",
                    right="right",
                    top="top",
                    bottom="bottom",
                    fill_color="fill_color",
                    line_color=None,
                    alpha=1 if i == 1 else 0.4,
                )
            else:
                self.glyph[i] = self.p.circle(
                    source=src, x="x", y="y", alpha=0.5, color="color"
                )
        self.update_selection(
            df,
            skewer_params,
            col_names,
            curr_selection,
        )

    def get_glyph_vars(self, ndim):
        list_glyph_vars = [
            ["x", "y", "text"],
            ["left", "right", "top", "bottom", "fill_color"],
            ["x", "y", "color"],
            ["left", "right", "top", "bottom", "fill_color"],
        ]
        return list_glyph_vars[ndim if ndim <= 3 else 3]

    def get_1d_hist_cds_dict(
        self, df, skewer_params, col_names, curr_selection, col_name_1d, for_nd=False
    ):
        df_1d_hist_combined = None
        nbins = 50
        if for_nd:
            col_name_1d_index = curr_selection["ndimplot_col_names"].index(col_name_1d)
        hist, edges = np.histogram(np.array(df[col_name_1d]), density=False, bins=nbins)
        for i, unique_val in enumerate(
            get_unique_vals(df, curr_selection["color_col_name"])
        ):
            df_unique_val = df[df[curr_selection["color_col_name"]] == unique_val]
            hist, edges = np.histogram(
                np.array(df_unique_val[col_name_1d]), density=False, bins=edges
            )
            df_1d_hist = pd.DataFrame(
                {
                    "bottom": range(nbins) if for_nd else edges[:-1],
                    "top": range(1, nbins + 1) if for_nd else edges[1:],
                    # "left": hist * (-1),
                    "count": hist,
                }
            )
            df_1d_hist["center"] = col_name_1d_index if for_nd else i
            df_1d_hist["fill_color"] = df_unique_val.iloc[0][
                skewer_params["color_col_name"]
            ]
            if not isinstance(df_1d_hist_combined, pd.DataFrame):
                df_1d_hist_combined = df_1d_hist
            else:
                df_1d_hist_combined = pd.concat(
                    [df_1d_hist_combined, df_1d_hist], ignore_index=True
                )
        df_1d_hist_combined["count_normalized"] = (
            df_1d_hist_combined["count"] / df_1d_hist_combined["count"].max()
        )
        df_1d_hist_combined["left"] = (
            df_1d_hist_combined["center"]
            - df_1d_hist_combined["count_normalized"] * 0.49
        )
        df_1d_hist_combined["right"] = (
            df_1d_hist_combined["center"]
            + df_1d_hist_combined["count_normalized"] * 0.49
        )

        return df_1d_hist_combined[self.get_glyph_vars(1)].to_dict("list")

    def get_scatterplot_cds_dict(self, df, skewer_params, col_names, col_x, col_y):
        df_scatterplot = df[[col_x, col_y, skewer_params["color_col_name"]]].rename(
            columns={
                col_x: "x",
                col_y: "y",
                skewer_params["color_col_name"]: "color",
            }
        )
        return df_scatterplot[self.get_glyph_vars(2)].to_dict("list")

    def get_nd_hist_cds_dict(self, df, skewer_params, col_names, curr_selection):
        df_nd_hist_combined = None
        for ndimplot_col_name in curr_selection["ndimplot_col_names"]:
            df_nd_hist = pd.DataFrame(
                self.get_1d_hist_cds_dict(
                    df,
                    skewer_params,
                    col_names,
                    curr_selection,
                    ndimplot_col_name,
                    for_nd=True,
                )
            )
            if not isinstance(df_nd_hist_combined, pd.DataFrame):
                df_nd_hist_combined = df_nd_hist
            else:
                df_nd_hist_combined = pd.concat(
                    [df_nd_hist_combined, df_nd_hist], ignore_index=True
                )

        return df_nd_hist_combined[self.get_glyph_vars(3)].to_dict("list")

    def clear_ndimplot(self):
        for i in range(4):
            self.glyph_vars = self.get_glyph_vars(i)
            self.glyph[i].data_source.data = to_empty_dict(self.glyph_vars)

    def update_selection(
        self, df, skewer_params, col_names, curr_selection, old_selection=None
    ):
        timer_obj = timer("Updating N-dim plot")
        self.clear_ndimplot()
        ndimplot_ndim = len(curr_selection["ndimplot_col_names"])
        if ndimplot_ndim == 0:
            self.glyph[0].data_source.data = {
                "x": [0],
                "y": [0],
                "text": ["No columns selected"],
            }
            self.p.xgrid.visible = False
            self.p.ygrid.visible = False
            self.p.xaxis.visible = False
            self.p.yaxis.visible = False
        elif ndimplot_ndim == 1:
            self.glyph[1].data_source.data = self.get_1d_hist_cds_dict(
                df,
                skewer_params,
                col_names,
                curr_selection,
                curr_selection["ndimplot_col_names"][0],
            )
            self.p.xgrid.visible = True
            self.p.ygrid.visible = False
            self.p.xaxis.visible = True
            self.p.yaxis.visible = True
            self.p.xaxis.axis_label = "Cluster ID"
            self.p.yaxis.axis_label = curr_selection["ndimplot_col_names"][0]
        elif ndimplot_ndim == 2:
            self.glyph[2].data_source.data = self.get_scatterplot_cds_dict(
                df,
                skewer_params,
                col_names,
                curr_selection["ndimplot_col_names"][0],
                curr_selection["ndimplot_col_names"][1],
            )
            self.p.xgrid.visible = True
            self.p.ygrid.visible = True
            self.p.xaxis.visible = True
            self.p.yaxis.visible = True
            self.p.xaxis.axis_label = curr_selection["ndimplot_col_names"][0]
            self.p.yaxis.axis_label = curr_selection["ndimplot_col_names"][1]
        else:  # ndimplot_ndim is 3 or above
            self.glyph[3].data_source.data = self.get_nd_hist_cds_dict(
                df,
                skewer_params,
                col_names,
                curr_selection,
            )
            self.p.xgrid.visible = True
            self.p.ygrid.visible = False
            self.p.xaxis.visible = True
            self.p.yaxis.visible = False
            self.p.xaxis.axis_label = "Dimensions"
            self.p.yaxis.axis_label = None

        timer_obj.done()

    def generate_ndimplot_figure(self, skewer_params):
        p = figure(
            width=skewer_params["widthspx_ndimplot"],
            height=skewer_params["heightspx_ndimplot"],
            tools=[
                BoxZoomTool(),
                PanTool(),
                WheelZoomTool(),
                ResetTool(),
                # LassoSelectTool(mode="append", select_every_mousemove=False),
            ],
            # title="N-dimensional plot (N >= 1)",
        )
        p.toolbar.logo = None
        p.min_border = 0
        return p


class mds_col_similarity_cl_membership:
    def __init__(self, skewer_params, col_names, dissimilarity_np):
        self.p = self.generate_figure(skewer_params)
        self.p.xaxis.major_label_text_font_size = "0pt"  # turn off x-axis tick labels
        self.p.yaxis.major_label_text_font_size = "0pt"  # turn off y-axis tick labels
        self.glyph_vars = ["x", "y", "color", "col_name"]
        self.glyph_vars_line = ["x1", "y1", "x2", "y2"]
        src = ColumnDataSource(to_empty_dict(self.glyph_vars))
        src_line = ColumnDataSource(to_empty_dict(self.glyph_vars_line))
        self.glyph_line = self.p.segment(
            source=src_line,
            x0="x1",
            y0="y1",
            x1="x2",
            y1="y2",
            line_color="silver",
            line_width=1,
        )
        self.glyph = self.p.circle(source=src, x="x", y="y", alpha=1, color="color")
        labels = LabelSet(
            source=src,
            x="x",
            y="y",
            text="col_name",
            x_offset=1,
            y_offset=1,
            level="underlay",
        )
        self.p.add_layout(labels)
        self.update_selection(
            skewer_params,
            col_names,
            dissimilarity_np,
        )

    def get_cds_dict(self, dissimilarity_np, col_names):
        pos = (
            MDS(
                random_state=4,
                eps=1e-9,
                max_iter=10000,
                dissimilarity="precomputed",
                normalized_stress="auto",
            )
            .fit(dissimilarity_np)
            .embedding_
        )
        df_scatterplot = pd.DataFrame(pos, columns=["x", "y"])
        df_scatterplot["color"] = "black"
        df_scatterplot["col_name"] = list(col_names)

        # # Create two DataFrames with shifted rows
        # df_shifted = df_scatterplot.shift(-1)  # Shifted one row up

        # # Filter out the last row which contains NaN due to shifting
        # df_shifted = df_shifted.dropna()

        # # Create a new DataFrame manually with consecutive connections
        # new_df = pd.DataFrame({
        #     'x1': df_scatterplot['x'].values[:-1],  # All rows except the last one
        #     'y1': df_scatterplot['y'].values[:-1],
        #     'x2': df_shifted['x'].values,  # All rows except the last one, shifted one position up
        #     'y2': df_shifted['y'].values
        # })

        # # Reset the index if desired
        # new_df.reset_index(drop=True, inplace=True)

        return df_scatterplot[self.glyph_vars].to_dict("list")

    def get_cds_dict2(self, dissimilarity_np, col_names):
        pos = (
            MDS(
                random_state=4,
                eps=1e-9,
                max_iter=10000,
                dissimilarity="precomputed",
                normalized_stress="auto",
            )
            .fit(dissimilarity_np)
            .embedding_
        )
        df_scatterplot = pd.DataFrame(pos, columns=["x", "y"])
        df_scatterplot["color"] = "black"
        df_scatterplot["col_name"] = list(col_names)

        # Create two DataFrames with shifted rows
        df_shifted = df_scatterplot.shift(-1)  # Shifted one row up

        # Filter out the last row which contains NaN due to shifting
        df_shifted = df_shifted.dropna()

        # Create a new DataFrame manually with consecutive connections
        new_df = pd.DataFrame(
            {
                "x1": df_scatterplot["x"].values[:-1],  # All rows except the last one
                "y1": df_scatterplot["y"].values[:-1],
                "x2": df_shifted[
                    "x"
                ].values,  # All rows except the last one, shifted one position up
                "y2": df_shifted["y"].values,
            }
        )

        # Reset the index if desired
        new_df.reset_index(drop=True, inplace=True)

        return new_df[self.glyph_vars_line].to_dict("list")

    def update_selection(self, skewer_params, col_names, dissimilarity_np):
        timer_obj = timer("Updating Scatterplot")
        self.glyph.data_source.data = self.get_cds_dict(dissimilarity_np, col_names)
        self.glyph_line.data_source.data = self.get_cds_dict2(
            dissimilarity_np, col_names
        )
        timer_obj.done()

    def generate_figure(self, skewer_params):
        p = figure(
            width=skewer_params["widthspx_mds_col_similarity_cl_membership"],
            height=skewer_params["heightspx_mds_col_similarity_cl_membership"],
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True,
        )
        p.toolbar.logo = None
        p.min_border = 0
        return p


class mds_nxn_setwise:
    def __init__(self, skewer_params, col_names, psets_vertical_ordering_df):
        self.p = self.generate_figure(skewer_params)
        self.glyph_vars = ["x", "y", "color", "label"]
        self.glyph_vars_wedge = ["x", "y", "color", "start_angle", "end_angle"]
        # self.glyph_vars_line = ["x1", "y1", "x2", "y2"]
        src = ColumnDataSource(to_empty_dict(self.glyph_vars))
        src_wedge = ColumnDataSource(to_empty_dict(self.glyph_vars_wedge))
        self.glyph_wedge = self.p.wedge(
            source=src,
            x="x",
            y="y",
            radius=0.05,
            fill_color="color",
            line_width=0,
            fill_alpha=0.2,
            start_angle="start_angle",
            end_angle="end_angle",
        )
        self.glyph = self.p.circle(
            source=src_wedge, x="x", y="y", alpha=1, color="color"
        )
        # labels = LabelSet(
        #     source=src,
        #     x="x",
        #     y="y",
        #     text="label",
        #     x_offset=1,
        #     y_offset=1,
        #     level="underlay",
        # )
        # self.p.add_layout(labels)
        self.update_selection(
            skewer_params,
            col_names,
            psets_vertical_ordering_df,
        )

    def get_cds_dict(self, psets_vertical_ordering_df, col_names):
        df_scatterplot = psets_vertical_ordering_df
        # df_scatterplot["col_name"] = list(col_names)

        # # Create two DataFrames with shifted rows
        # df_shifted = df_scatterplot.shift(-1)  # Shifted one row up

        # # Filter out the last row which contains NaN due to shifting
        # df_shifted = df_shifted.dropna()

        # # Create a new DataFrame manually with consecutive connections
        # new_df = pd.DataFrame({
        #     'x1': df_scatterplot['x'].values[:-1],  # All rows except the last one
        #     'y1': df_scatterplot['y'].values[:-1],
        #     'x2': df_shifted['x'].values,  # All rows except the last one, shifted one position up
        #     'y2': df_shifted['y'].values
        # })

        # # Reset the index if desired
        # new_df.reset_index(drop=True, inplace=True)

        return df_scatterplot[self.glyph_vars].to_dict("list")

    def get_cds_dict_wedge(self, psets_vertical_ordering_df, col_names):
        # df_scatterplot = psets_vertical_ordering_df
        df_scatterplot_wedge = copy.deepcopy(psets_vertical_ordering_df)

        # print(psets_vertical_ordering_df)
        df_scatterplot_wedge["start_angle"] = df_scatterplot_wedge.apply(
            lambda row: (
                col_names.index(row["partition_col_name"]) * 2 * np.pi / len(col_names)
            )
            + np.pi / 2,
            axis=1,
        )
        df_scatterplot_wedge["end_angle"] = df_scatterplot_wedge.apply(
            lambda row: (
                (col_names.index(row["partition_col_name"]) + 1)
                * 2
                * np.pi
                / len(col_names)
            )
            + np.pi / 2,
            axis=1,
        )
        print(df_scatterplot_wedge)
        return df_scatterplot_wedge[self.glyph_vars_wedge].to_dict("list")

    # def get_cds_dict2(self, dissimilarity_np, col_names):
    #     pos = (
    #         MDS(
    #             random_state=4,
    #             eps=1e-9,
    #             max_iter=10000,
    #             dissimilarity="precomputed",
    #             normalized_stress="auto",
    #         )
    #         .fit(dissimilarity_np)
    #         .embedding_
    #     )
    #     df_scatterplot = pd.DataFrame(pos, columns=["x", "y"])
    #     df_scatterplot["color"] = "black"
    #     df_scatterplot["col_name"] = list(col_names)

    #     # Create two DataFrames with shifted rows
    #     df_shifted = df_scatterplot.shift(-1)  # Shifted one row up

    #     # Filter out the last row which contains NaN due to shifting
    #     df_shifted = df_shifted.dropna()

    #     # Create a new DataFrame manually with consecutive connections
    #     new_df = pd.DataFrame(
    #         {
    #             "x1": df_scatterplot["x"].values[:-1],  # All rows except the last one
    #             "y1": df_scatterplot["y"].values[:-1],
    #             "x2": df_shifted[
    #                 "x"
    #             ].values,  # All rows except the last one, shifted one position up
    #             "y2": df_shifted["y"].values,
    #         }
    #     )

    #     # Reset the index if desired
    #     new_df.reset_index(drop=True, inplace=True)

    #     return new_df[self.glyph_vars_line].to_dict("list")

    def update_selection(self, skewer_params, col_names, psets_vertical_ordering_df):
        timer_obj = timer("Updating Scatterplot")
        self.glyph.data_source.data = self.get_cds_dict(
            psets_vertical_ordering_df, col_names
        )
        self.glyph_wedge.data_source.data = self.get_cds_dict_wedge(
            psets_vertical_ordering_df, col_names
        )
        timer_obj.done()

    def generate_figure(self, skewer_params):
        p = figure(
            width=skewer_params["widthspx_mds_col_similarity_cl_membership"],
            height=skewer_params["heightspx_mds_col_similarity_cl_membership"],
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True,
        )
        p.toolbar.logo = None
        p.min_border = 0
        p.xaxis.major_label_text_font_size = "0pt"  # turn off x-axis tick labels
        p.yaxis.major_label_text_font_size = "0pt"  # turn off y-axis tick labels
        return p


class similarity_roof_shaped_matrix_diagram:
    def __init__(self, skewer_params, col_names, dissimilarity_np):
        data_table_src = ColumnDataSource(
            self.get_half_matrix_df(col_names, dissimilarity_np)
        )
        self.data_table = DataTable(
            source=data_table_src,
            columns=[
                TableColumn(field="col_pair", title="Pair of Columns", sortable=False),
                # TableColumn(field="col_name_0", title="Column Name 1", sortable=False),
                # TableColumn(field="col_name_1", title="Column Name 2", sortable=False),
                TableColumn(field="dissimilarity", title="Dissimilarity Measure"),
                TableColumn(field="bar_chart", title="Bar Chart", sortable=False),
            ],
            width=skewer_params["widthspx_dissimilarity_data_table"],
            height=skewer_params["heightspx_dissimilarity_data_table"],
        )

        self.p = self.generate_figure(skewer_params, col_names)
        self.cell_glyph_vars = ["xs", "ys"]
        cell_src = ColumnDataSource(to_empty_dict(self.cell_glyph_vars))
        self.cell_glyph = self.p.patches(
            source=cell_src,
            xs="xs",
            ys="ys",
            fill_color="gray",
            line_color="black",
            line_width=1,
        )
        self.segment_glyph_vars = ["xs", "ys"]
        segment_src = ColumnDataSource(to_empty_dict(self.segment_glyph_vars))
        self.segment_glyph = self.p.multi_line(
            source=segment_src,
            xs="xs",
            ys="ys",
            line_color="black",
            line_width=1,
        )
        self.ray_glyph_vars = ["x", "y"]
        ray_src = ColumnDataSource(to_empty_dict(self.ray_glyph_vars))
        self.ray_glyph = self.p.ray(
            source=ray_src,
            x="x",
            y="y",
            length=0,
            angle=np.pi,
            line_color="black",
            line_width=1,
        )
        self.update_selection(
            skewer_params,
            col_names,
            dissimilarity_np,
        )

    def get_half_matrix_df(self, col_names, dissimilarity_np):
        df_cell_0 = pd.DataFrame(
            {"col_name_0": list(col_names), "col_id_0": range(len(col_names))}
        )
        df_cell_1 = pd.DataFrame(
            {"col_name_1": list(col_names), "col_id_1": range(len(col_names))}
        )
        df_rect = df_cell_0.merge(df_cell_1, how="cross")
        df_rect = df_rect[df_rect["col_id_0"] < df_rect["col_id_1"]]
        df_rect["dissimilarity"] = df_rect.apply(
            lambda row: dissimilarity_np[row["col_id_0"]][row["col_id_1"]],
            axis=1,
        )
        df_rect["bar_chart"] = df_rect.apply(
            lambda row: number_to_text_bar_chart(row["dissimilarity"], 14),
            axis=1,
        )
        df_rect["col_pair"] = df_rect.apply(
            lambda row: row["col_name_0"] + " - " + row["col_name_1"],
            axis=1,
        )
        return df_rect

    def get_cell_cds_dict(self, col_names, dissimilarity_np):
        df_rect = self.get_half_matrix_df(col_names, dissimilarity_np)
        df_rect["cx"] = (df_rect["col_id_1"] - df_rect["col_id_0"]) / 2
        df_rect["cy"] = (df_rect["col_id_0"] + df_rect["col_id_1"]) / 2
        half_diagonal_length = 0.5
        df_rect["xs"] = df_rect.apply(
            lambda row: [
                row["cx"] - half_diagonal_length * row["dissimilarity"],
                row["cx"],
                row["cx"] + half_diagonal_length * row["dissimilarity"],
                row["cx"],
            ],
            axis=1,
        )
        df_rect["ys"] = df_rect.apply(
            lambda row: [
                row["cy"],
                row["cy"] - half_diagonal_length * row["dissimilarity"],
                row["cy"],
                row["cy"] + half_diagonal_length * row["dissimilarity"],
            ],
            axis=1,
        )
        return df_rect[self.cell_glyph_vars].to_dict("list")

    def get_segment_cds_dict(self, col_names):
        start_x = 0
        df_segment = pd.DataFrame(
            {
                "col_name": list(col_names)[:-1],
                "col_id": range(len(col_names) - 1),
            }
        )
        df_segment["xs"] = df_segment.apply(
            lambda row: [
                start_x,
                0,
                (len(col_names) - row["col_id"]) * 0.5,
                (len(col_names) - row["col_id"] - 1) * 0.5,
                0,
                start_x,
            ],
            axis=1,
        )
        df_segment["ys"] = df_segment.apply(
            lambda row: [
                row["col_id"] - 0.5,
                row["col_id"] - 0.5,
                (len(col_names) + row["col_id"] - 1) / 2,
                (len(col_names) + row["col_id"]) / 2,
                row["col_id"] + 0.5,
                row["col_id"] + 0.5,
            ],
            axis=1,
        )
        df_segment_0 = df_segment
        df_segment = pd.DataFrame(
            {
                "col_name": list(col_names)[1:],
                "col_id": range(1, len(col_names)),
            }
        )
        df_segment["xs"] = df_segment.apply(
            lambda row: [
                start_x,
                0,
                (row["col_id"]) * 0.5,
                (row["col_id"] + 1) * 0.5,
                0,
                start_x,
            ],
            axis=1,
        )
        df_segment["ys"] = df_segment.apply(
            lambda row: [
                row["col_id"] - 0.5,
                row["col_id"] - 0.5,
                (row["col_id"] - 1) / 2,
                (row["col_id"]) / 2,
                row["col_id"] + 0.5,
                row["col_id"] + 0.5,
            ],
            axis=1,
        )
        df_segment = pd.concat([df_segment_0, df_segment], axis=0)
        return df_segment[self.segment_glyph_vars].to_dict("list")

    def get_ray_cds_dict(self, col_names):
        df_ray = pd.DataFrame(
            {
                "col_name": list(col_names),
                "col_id": range(len(col_names)),
            }
        )
        df_ray["x"] = 0
        df_ray["y"] = df_ray["col_id"] - 0.5
        df_ray_0 = df_ray
        df_ray = pd.DataFrame(
            {
                "col_name": list(col_names),
                "col_id": range(len(col_names)),
            }
        )
        df_ray["x"] = 0
        df_ray["y"] = df_ray["col_id"] + 0.5

        df_ray = pd.concat([df_ray_0, df_ray], axis=0)
        return df_ray[self.ray_glyph_vars].to_dict("list")

    def update_selection(self, skewer_params, col_names, dissimilarity_np):
        # timer_obj = timer("Updating Scatterplot")
        # self.labels.source.data = self.get_labels_cds_dict(col_names)
        self.cell_glyph.data_source.data = self.get_cell_cds_dict(
            col_names, dissimilarity_np
        )
        self.segment_glyph.data_source.data = self.get_segment_cds_dict(col_names)
        self.ray_glyph.data_source.data = self.get_ray_cds_dict(col_names)
        tick_labels = {}
        for col_id, col_name in enumerate(col_names):
            tick_labels[col_id] = col_name
        self.p.yaxis.ticker = list(range(len(col_names)))
        self.p.yaxis.major_label_overrides = tick_labels
        # timer_obj.done()

    def generate_figure(self, skewer_params, col_names):
        p = figure(
            width=skewer_params["widthspx_similarity_roof_shaped_matrix_diagram"],
            height=skewer_params["heightspx_similarity_roof_shaped_matrix_diagram"],
            tools="",
        )
        p.toolbar.logo = None
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.xaxis.visible = False
        p.x_range.range_padding = 0.2
        p.y_range.flipped = True
        p.outline_line_color = None
        p.min_border = 0
        return p
