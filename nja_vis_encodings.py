import numpy as np
import pandas as pd
import copy
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.manifold import MDS
from bokeh.plotting import figure
from bokeh.models import (
    Span,
    LabelSet,
    ColumnDataSource,
    Range1d,
)
from bokeh.palettes import turbo

from nja_preprocessing import get_color_sort_order
from nja_helper_functions import to_empty_dict, timer


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


class alluvial:
    def __init__(
        self,
        df,
        psets_vertical_ordering_df,
        psets_color,
        nja_params,
        col_names,
        curr_selection,
    ):
        self.p = self.generate_figure(nja_params, col_names)
        draw_vlines(self.p, col_names)
        timer_obj = timer("Updating Alluvial Diagram")
        self.alluvial_cluster_bars_obj = self.alluvial_cluster_bars(
            self.p,
            df,
            nja_params,
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
            nja_params,
            col_names,
            curr_selection,
        )
        self.rb_obj = self.radio_buttons(
            self.p,
            nja_params,
            col_names,
            curr_selection,
        )
        timer_obj.done()

    class alluvial_cluster_bars:
        def __init__(
            self,
            p,
            df,
            nja_params,
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
                nja_params,
                col_names,
                psets_vertical_ordering_df,
                curr_selection,
            )

        def get_cds_dict(
            self,
            df,
            nja_params,
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
                df_cb["width"] *= nja_params["width_per_count"]
                df_cb["col_name"] = col_name
                df_cb["left"] = col_names.get_col_id(col_name) - nja_params["bar_width"]
                df_cb["right"] = (
                    col_names.get_col_id(col_name) + nja_params["bar_width"]
                )
                df_cb["line_color"] = df_cb.apply(
                    lambda row: self.get_line_color(
                        df,
                        nja_params,
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
                if (
                    curr_selection["color_col_name"] == None
                    or curr_selection["color_col_name"] == col_name
                ):
                    df_cb["fill_color"] = df_cb["line_color"]
                    df_cb["fill_alpha"] = 1.0
                else:
                    df_cb["fill_color"] = None
                    df_cb["fill_alpha"] = 0.0
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
            nja_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
            old_selection=None,
        ):
            self.glyph.data_source.data = self.get_cds_dict(
                df,
                nja_params,
                col_names,
                psets_vertical_ordering_df,
                curr_selection,
            )

        def get_line_color(
            self,
            df,
            nja_params,
            psets_vertical_ordering_df,
            curr_selection,
            col_name,
            cluster_id,
        ):
            if (
                col_name != curr_selection["color_col_name"]
                and curr_selection["color_col_name"] != None
            ):
                return nja_params["cluster_bars_default_line_color"]
            if (
                len(curr_selection["cluster_ids"]) == 0
                or cluster_id in curr_selection["cluster_ids"]
            ) and curr_selection["color_col_name"] != None:
                return (df[df[col_name] == cluster_id].iloc[0])[
                    nja_params["color_col_name"]
                ]
            if curr_selection["color_col_name"] == None:
                return psets_vertical_ordering_df[
                    psets_vertical_ordering_df["label"] == (col_name, cluster_id)
                ]["color"].values[0]
            return nja_params["cluster_bars_filtered_out_line_color"]

    class alluvial_edges:
        def __init__(
            self,
            p,
            df,
            df_filtered,
            psets_vertical_ordering_df,
            psets_color,
            nja_params,
            col_names,
            curr_selection,
        ):
            self.resolution = 100
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
            )
            self.update_selection(
                df,
                df_filtered,
                psets_vertical_ordering_df,
                psets_color,
                nja_params,
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
                ###############################################################################################
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
            nja_params,
            col_names,
            curr_selection,
            old_selection=None,
        ):
            self.glyph.data_source.data = self.get_cds_dict(
                df,
                df_filtered,
                col_names,
                nja_params["color_col_name"],
                nja_params["width_per_count"],
                psets_vertical_ordering_df,
                psets_color,
                nja_params["bar_width"],
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
        def __init__(self, p, nja_params, col_names, curr_selection):
            self.glyphs = {}
            self.glyphs["colorful_boundary"] = self.rb_colorful_boundary(
                p,
                nja_params,
                col_names,
                curr_selection,
            )
            self.glyphs["elliptical_buttons"] = self.rb_elliptical_buttons(
                p,
                nja_params,
                col_names,
                curr_selection,
            )
            self.glyphs["button_labels"] = self.rb_button_labels(
                p,
                nja_params,
                col_names,
                curr_selection,
            )

        def update_selection(
            self, nja_params, col_names, curr_selection, old_selection
        ):
            for glyph_type in self.glyphs.keys():
                self.glyphs[glyph_type].update_selection(
                    nja_params, col_names, curr_selection, old_selection
                )

        class rb_elliptical_buttons:
            def __init__(self, p, nja_params, col_names, curr_selection):
                self.glyph_vars = ["x", "fill_color", "hatch_pattern"]
                src = ColumnDataSource(to_empty_dict(self.glyph_vars))
                self.glyph = p.ellipse(
                    source=src,
                    x="x",
                    y=nja_params["rb_ellipse_y"],
                    width=nja_params["rb_ellipse_width"],
                    height=nja_params["rb_ellipse_height"],
                    fill_color="fill_color",
                    line_color=nja_params["rb_line_color"],
                    line_width=nja_params["rb_line_width"],
                    hatch_color=nja_params["rb_line_color"],
                    hatch_pattern="hatch_pattern",
                    hatch_alpha=0.5,
                    hatch_scale=5,
                )
                self.update_selection(nja_params, col_names, curr_selection)

            def get_cds_dict(self, nja_params, col_names, curr_selection):
                df = pd.DataFrame(col_names, columns=["col_names"])
                df["x"] = df.apply(
                    lambda row: col_names.get_col_id(row["col_names"]),
                    axis=1,
                )
                df["fill_color"] = df.apply(
                    lambda row: (
                        nja_params["rb_fill_color_selected"]
                        if row["col_names"] == curr_selection["color_col_name"]
                        else nja_params["rb_fill_color_unselected"]
                    ),
                    axis=1,
                )
                df["hatch_pattern"] = df.apply(
                    lambda row: (
                        nja_params["rb_hatch_pattern_filtered_column"]
                        if (
                            row["col_names"] == curr_selection["color_col_name"]
                            and len(curr_selection["cluster_ids"]) != 0
                        )
                        else nja_params["rb_hatch_pattern_not_filtered_column"]
                    ),
                    axis=1,
                )
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, nja_params, col_names, curr_selection, old_selection=None
            ):
                self.glyph.data_source.data = self.get_cds_dict(
                    nja_params, col_names, curr_selection
                )

        class rb_button_labels:
            def __init__(self, p, nja_params, col_names, curr_selection):
                self.glyph_vars = ["x", "text", "text_font_size", "text_color"]
                src = ColumnDataSource(to_empty_dict(self.glyph_vars))
                self.glyph = LabelSet(
                    source=src,
                    x="x",
                    y=nja_params["rb_labels_y"],
                    text="text",
                    text_font_size="text_font_size",
                    text_align="center",
                    text_font_style="bold",
                    text_color="text_color",
                )
                p.add_layout(self.glyph)
                self.update_selection(nja_params, col_names, curr_selection)

            def get_cds_dict(self, nja_params, col_names, curr_selection):
                df = pd.DataFrame(col_names, columns=["col_names"])
                df["x"] = df.apply(
                    lambda row: col_names.get_col_id(row["col_names"]),
                    axis=1,
                )
                # TODO: Handle larger strings and numbers
                # for eg. ("{:.1E}".format(val) if len(str(val)) >= 5 else str(val))
                df["text"] = df["col_names"]
                df["text_font_size"] = df.apply(
                    lambda row: "12pt" if len(row["text"]) > 3 else "12pt",
                    axis=1,
                )
                df["text_color"] = df.apply(
                    lambda row: (
                        nja_params["rb_fill_color_unselected"]
                        if row["col_names"] == curr_selection["color_col_name"]
                        else nja_params["rb_fill_color_selected"]
                    ),
                    axis=1,
                )
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, nja_params, col_names, curr_selection, old_selection=None
            ):
                self.glyph.source.data = self.get_cds_dict(
                    nja_params, col_names, curr_selection
                )

        class rb_colorful_boundary:
            def __init__(self, p, nja_params, col_names, curr_selection):
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
                self.update_selection(nja_params, col_names, curr_selection)

            def get_cds_dict(self, nja_params, center_x):
                if center_x == None:
                    return to_empty_dict(self.glyph_vars)
                center_y = nja_params["rb_ellipse_y"]
                angle_linspace = np.linspace(
                    0 + np.pi / 2, 2 * np.pi + np.pi / 2, num=self.n_colors + 1
                )
                df = pd.DataFrame(
                    list(zip(angle_linspace[:-1], angle_linspace[1:])),
                    columns=["angle0", "angle1"],
                )
                df["x0"] = df.apply(
                    lambda row: nja_params["rb_ellipse_bondary_halfwidth"]
                    * np.cos(row["angle0"])
                    + center_x,
                    axis=1,
                )
                df["x1"] = df.apply(
                    lambda row: nja_params["rb_ellipse_bondary_halfwidth"]
                    * np.cos(row["angle1"])
                    + center_x,
                    axis=1,
                )
                df["y0"] = df.apply(
                    lambda row: nja_params["rb_ellipse_bondary_halfheight"]
                    * np.sin(row["angle0"])
                    + center_y,
                    axis=1,
                )
                df["y1"] = df.apply(
                    lambda row: nja_params["rb_ellipse_bondary_halfheight"]
                    * np.sin(row["angle1"])
                    + center_y,
                    axis=1,
                )
                df["line_color"] = list(turbo(self.n_colors))
                return df[self.glyph_vars].to_dict("list")

            def update_selection(
                self, nja_params, col_names, curr_selection, old_selection=None
            ):
                center_x = (
                    col_names.get_col_id(curr_selection["color_col_name"])
                    if curr_selection["color_col_name"] != None
                    else None
                )
                self.glyph.data_source.data = self.get_cds_dict(nja_params, center_x)

    def update_selection(
        self,
        df,
        df_filtered,
        psets_vertical_ordering_df,
        psets_color,
        nja_params,
        col_names,
        curr_selection,
        old_selection,
    ):
        timer_obj = timer("Updating Alluvial Diagram")
        self.rb_obj.update_selection(
            nja_params, col_names, curr_selection, old_selection
        )
        self.alluvial_edges_obj.update_selection(
            df,
            df_filtered,
            psets_vertical_ordering_df,
            psets_color,
            nja_params,
            col_names,
            curr_selection,
            old_selection,
        )
        self.alluvial_cluster_bars_obj.update_selection(
            df,
            nja_params,
            col_names,
            psets_vertical_ordering_df,
            curr_selection,
            old_selection,
        )
        timer_obj.done()

    def generate_figure(self, nja_params, col_names):
        p = figure(
            width=nja_params["widthspx_alluvial"],
            height=nja_params["heightspx_alluvial"],
            tools="",
            y_range=(
                nja_params["alluvial_y_start"],
                nja_params["alluvial_y_end"],
            ),
        )
        p.toolbar.logo = None
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.yaxis.axis_line_color = None
        p.outline_line_color = None
        p.min_border = 0
        p.x_range = Range1d(-0.5, len(col_names) - 0.5)
        p.yaxis.major_tick_line_color = "lightgray"

        return p


class ixn_merge_split:
    def __init__(
        self,
        x_range,
        df,
        nja_params,
        psets_color,
        col_names,
        curr_selection,
    ):
        self.p_normal, self.p_inverted = self.generate_figures(x_range, nja_params)
        draw_vlines(self.p_normal, col_names)
        draw_vlines(self.p_inverted, col_names)
        self.draw_axis_line()
        self.p_normal.yaxis.axis_label = "Merge Measure"
        self.merge_measure = self.cim_merge(
            self.p_normal,
            x_range,
            df,
            nja_params,
            psets_color,
            col_names,
            curr_selection,
        )
        self.p_inverted.yaxis.axis_label = "Split Measure"
        self.split_measure = self.cim_split(
            self.p_inverted,
            df,
            nja_params,
            col_names,
            curr_selection,
        )

    class cim_merge:
        def __init__(
            self,
            p,
            x_range,
            df,
            nja_params,
            psets_color,
            col_names,
            curr_selection,
        ):
            self.glyph_vars = ["left", "right", "top", "bottom"]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.quad(
                source=src,
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                line_color=None,
                fill_color="darkgray",
                fill_alpha=1,
            )
            self.update_selection(
                df,
                df,
                nja_params,
                psets_color,
                col_names,
                curr_selection,
            )

        def get_cds_dict(
            self,
            df_non_filtered,
            df,
            nja_params,
            psets_color,
            col_names,
            wrt_col_name,
            bool_measure_only=False,
        ):
            if not bool_measure_only:
                if wrt_col_name == None:
                    return to_empty_dict(self.glyph_vars)
                df_cim_merge_combined = None
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
                df_cim_merge = (
                    df[[col_name, wrt_col_name]]
                    .groupby([col_name, wrt_col_name])
                    .agg(
                        count=(col_name, "size"),
                    )
                )
                df_cim_merge_temp = (
                    df_cim_merge[["count"]]
                    .groupby(level=0)
                    .sum()
                    .rename(columns={"count": "count_total"})
                )
                df_cim_merge = df_cim_merge.join(
                    df_cim_merge_temp.reindex(df_cim_merge.index, level=0)
                )
                df_cim_merge["cim_merge_individual"] = (
                    (df_cim_merge["count_total"] - df_cim_merge["count"])
                    * df_cim_merge["count"]
                ) / df_cim_merge["count_total"]

                df_cim_merge["cim_merge"] = (
                    df_cim_merge["cim_merge_individual"] / df_total_count
                )

                if bool_measure_only:
                    dissimilarity_dict[(wrt_col_name, col_name)] = df_cim_merge[
                        "cim_merge"
                    ].sum()
                    continue

                df_cim_merge = (
                    df_cim_merge.reset_index()
                    .sort_values([wrt_col_name], ascending=[True])
                    .set_index([wrt_col_name, col_name])
                )

                df_cim_merge_temp = (
                    df_cim_merge[["cim_merge", "cim_merge_individual", "count"]]
                    .groupby(level=0)
                    .agg(
                        cim_merge=("cim_merge", "sum"),
                        cim_merge_individual=("cim_merge_individual", "sum"),
                        count=("count", "sum"),
                        # fill_color=("fill_color", "first"),
                    )
                )
                df_cim_merge_temp["width_percentage"] = 1
                df_cim_merge_temp["bar_halfwidth"] = df_cim_merge_temp[
                    "width_percentage"
                ] * (nja_params["cim_bar_width"] / 2)
                df_cim_merge_temp["left"] = (
                    col_names.get_col_id(col_name) - df_cim_merge_temp["bar_halfwidth"]
                )
                df_cim_merge_temp["right"] = (
                    col_names.get_col_id(col_name) + df_cim_merge_temp["bar_halfwidth"]
                )
                df_cim_merge_temp["top"] = df_cim_merge_temp["cim_merge"].sum()
                df_cim_merge_temp["bottom"] = 0
                df_cim_merge_temp["fill_color"] = "gray"

                df_cim_merge = df_cim_merge_temp

                if not isinstance(df_cim_merge_combined, pd.DataFrame):
                    df_cim_merge_combined = df_cim_merge
                else:
                    df_cim_merge_combined = pd.concat(
                        [df_cim_merge_combined, df_cim_merge], ignore_index=True
                    )
            if bool_measure_only:
                return dissimilarity_dict
            return df_cim_merge_combined[self.glyph_vars].to_dict(
                "list"
            )  # , df_cim_merge_background[self.background_glyph_vars].to_dict("list")

        def update_selection(
            self,
            df_non_filtered,
            df,
            nja_params,
            psets_color,
            col_names,
            curr_selection,
            old_selection=None,
        ):
            timer_obj = timer("Updating Merge Measure")
            self.glyph.data_source.data = self.get_cds_dict(
                df_non_filtered,
                df,
                nja_params,
                psets_color,
                col_names,
                curr_selection["color_col_name"],
            )
            timer_obj.done()

    class cim_split:
        def __init__(self, p, df, nja_params, col_names, curr_selection):
            self.glyph_vars = ["left", "right", "top", "bottom"]
            src = ColumnDataSource(to_empty_dict(self.glyph_vars))
            self.glyph = p.quad(
                source=src,
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                line_width=0,
                fill_color="darkgray",
                fill_alpha=1,
            )
            self.update_selection(
                df,
                nja_params,
                col_names,
                curr_selection,
            )

        def get_cds_dict(self, df, nja_params, col_names, wrt_col_name):
            if wrt_col_name == None:
                return to_empty_dict(self.glyph_vars)
            df_cim_split_combined = None
            df_total_count = len(df.index)
            df_cim_split_wrt_col_name = (
                df[[wrt_col_name]]
                .groupby([wrt_col_name])
                .agg(
                    count=(wrt_col_name, "size"),
                )
                .reset_index()
                .set_index([wrt_col_name])
            )
            population_count = len(df.index)

            for col_name in col_names:
                if col_name == wrt_col_name:
                    continue
                df_cim_split = (
                    df[
                        [
                            wrt_col_name,
                            col_name,
                        ]
                    ]
                    .groupby([wrt_col_name, col_name])
                    .agg(
                        count_intersection=(col_name, "size"),
                    )
                    .reset_index()
                    .set_index([wrt_col_name, col_name])
                )
                df_cim_split["count_selection_set"] = df_cim_split.apply(
                    lambda row: df_cim_split_wrt_col_name.loc[(row.name[0]), "count"],
                    axis=1,
                )
                df_cim_split["gini"] = df_cim_split.apply(
                    lambda row: (row["count_intersection"] * row["count_intersection"])
                    / (row["count_selection_set"] * row["count_selection_set"]),
                    axis=1,
                )
                df_cim_split = (
                    df_cim_split.reset_index()
                    .groupby([wrt_col_name])
                    .agg(
                        gini=("gini", "sum"),
                        count_selection_set=("count_selection_set", "first"),
                    )
                )
                df_cim_split["gini"] = 1 - df_cim_split["gini"]
                df_cim_split["cim_split_measure"] = (
                    df_cim_split["gini"]
                    * df_cim_split["count_selection_set"]
                    / population_count
                )
                df_cim_split = pd.DataFrame(
                    {
                        "left": [
                            col_names.get_col_id(col_name)
                            - nja_params["cim_bar_width"] / 2
                        ],
                        "right": [
                            col_names.get_col_id(col_name)
                            + nja_params["cim_bar_width"] / 2
                        ],
                        "top": df_cim_split["cim_split_measure"].sum(),
                        "bottom": 0,
                    }
                )
                if not isinstance(df_cim_split_combined, pd.DataFrame):
                    df_cim_split_combined = df_cim_split
                else:
                    df_cim_split_combined = pd.concat(
                        [df_cim_split_combined, df_cim_split], ignore_index=True
                    )
            return df_cim_split_combined[self.glyph_vars].to_dict("list")

        def update_selection(
            self, df, nja_params, col_names, curr_selection, old_selection=None
        ):
            timer_obj = timer("Updating Split Measure")
            self.glyph.data_source.data = self.get_cds_dict(
                df, nja_params, col_names, curr_selection["color_col_name"]
            )
            timer_obj.done()

    def update_selection(
        self,
        df_non_filtered,
        df_filtered,
        nja_params,
        psets_color,
        col_names,
        curr_selection,
        old_selection,
    ):
        self.merge_measure.update_selection(
            df_non_filtered,
            df_filtered,
            nja_params,
            psets_color,
            col_names,
            curr_selection,
            old_selection,
        )
        self.split_measure.update_selection(
            df_filtered, nja_params, col_names, curr_selection, old_selection
        )

    def generate_figure(self, x_range, nja_params):
        if x_range != None:
            p = figure(
                width=nja_params["widthspx_ixn_merge_split"],
                height=nja_params["heightspx_ixn_merge_split"],
                x_range=x_range,
                tools="",
            )
        else:
            p = figure(
                width=nja_params["widthspx_ixn_merge_split"],
                height=nja_params["heightspx_ixn_merge_split"],
                tools="",
            )
        p.toolbar.logo = None
        p.xgrid.visible = False
        p.xaxis.visible = False
        p.outline_line_color = None
        p.min_border = 0
        p.xaxis.major_label_text_font_size = "0pt"
        p.x_range = x_range
        return p

    def generate_figures(self, x_range, nja_params):
        p_normal = self.generate_figure(x_range, nja_params)
        p_inverted = self.generate_figure(x_range, nja_params)
        p_normal.y_range.start = 0.001
        p_normal.y_range.end = 1.01
        p_inverted.y_range.start = 1.01
        p_inverted.y_range.end = 0.001
        p_inverted.y_range.flipped = True
        return p_normal, p_inverted

    def draw_axis_line(self):
        axis_line = Span(
            location=0,
            dimension="width",
            line_color="black",
            line_width=3,
            level="overlay",
        )
        self.p_normal.renderers.extend([axis_line])
        self.p_inverted.renderers.extend([axis_line])


class nxn_sets:
    def __init__(
        self,
        df,
        nja_params,
        col_names,
        curr_selection,
        psets_vertical_ordering_df,
    ):
        self.p = self.generate_figure(nja_params)
        self.glyph_vars = ["x", "y", "color", "label", "fill_alpha"]
        self.glyph_vars_segment = ["x", "y", "y1", "color", "line_alpha"]
        self.glyph_vars_wedge = [
            "x",
            "y",
            "color",
            "start_angle",
            "end_angle",
            "fill_alpha",
        ]
        src = ColumnDataSource(to_empty_dict(self.glyph_vars))
        src_segment = ColumnDataSource(to_empty_dict(self.glyph_vars_segment))
        src_wedge = ColumnDataSource(to_empty_dict(self.glyph_vars_wedge))

        self.glyph = self.p.circle(
            source=src, x="x", y="y", alpha="fill_alpha", color="color"
        )
        self.glyph_segment = self.p.segment(
            source=src_segment,
            x0="x",
            y0="y",
            x1="x",
            y1="y1",
            line_color="color",
            line_alpha="line_alpha",
        )
        self.glyph_wedge = self.p.wedge(
            source=src_wedge,
            x="x",
            y="y",
            radius=0.05,
            fill_color="color",
            line_width=0,
            fill_alpha="fill_alpha",
            start_angle="start_angle",
            end_angle="end_angle",
            direction="clock",
        )
        self.update_selection(
            df,
            df,
            nja_params,
            col_names,
            curr_selection,
            psets_vertical_ordering_df,
        )

    def get_cds_dict(
        self, df_filtered, psets_vertical_ordering_df, col_names, curr_selection
    ):
        df_scatterplot = copy.deepcopy(psets_vertical_ordering_df)
        df_scatterplot["fill_alpha"] = df_scatterplot.apply(
            lambda row: (
                1
                if (
                    df_filtered[row["partition_col_name"]]
                    == row["partition_set_categorical_value"]
                ).any()
                else 0
            ),
            axis=1,
        )
        return df_scatterplot[self.glyph_vars].to_dict("list")

    def get_cds_dict_segment(
        self, df_filtered, psets_vertical_ordering_df, col_names, curr_selection
    ):
        df_scatterplot = copy.deepcopy(psets_vertical_ordering_df)
        df_scatterplot["y1"] = df_scatterplot["y"] + 0.05
        # print(col_names[len(col_names) - 1])
        df_scatterplot["line_alpha"] = df_scatterplot.apply(
            lambda row: (
                0.7
                if (
                    df_filtered[row["partition_col_name"]]
                    == row["partition_set_categorical_value"]
                ).any()
                and row["partition_col_name"] == col_names[len(col_names) - 1]
                else 0
            ),
            axis=1,
        )
        return df_scatterplot[self.glyph_vars_segment].to_dict("list")

    def get_cds_dict_wedge(
        self, df_filtered, psets_vertical_ordering_df, col_names, curr_selection
    ):
        df_scatterplot_wedge = copy.deepcopy(psets_vertical_ordering_df)
        df_scatterplot_wedge["start_angle"] = df_scatterplot_wedge.apply(
            lambda row: (
                -col_names.index(row["partition_col_name"]) * 2 * np.pi / len(col_names)
            )
            + np.pi / 2,
            axis=1,
        )
        df_scatterplot_wedge["end_angle"] = df_scatterplot_wedge.apply(
            lambda row: (
                -(col_names.index(row["partition_col_name"]) + 1)
                * 2
                * np.pi
                / len(col_names)
            )
            + np.pi / 2,
            axis=1,
        )
        df_scatterplot_wedge["fill_alpha"] = df_scatterplot_wedge.apply(
            lambda row: (
                1
                if row["partition_col_name"] == curr_selection["color_col_name"]
                else 0.2
            ),
            axis=1,
        )
        df_scatterplot_wedge["fill_alpha"] = df_scatterplot_wedge.apply(
            lambda row: (
                row["fill_alpha"]
                if (
                    df_filtered[row["partition_col_name"]]
                    == row["partition_set_categorical_value"]
                ).any()
                else 0
            ),
            axis=1,
        )
        return df_scatterplot_wedge[self.glyph_vars_wedge].to_dict("list")

    def update_selection(
        self,
        df,
        df_filtered,
        nja_params,
        col_names,
        curr_selection,
        psets_vertical_ordering_df,
    ):
        timer_obj = timer("Updating NxN Sets")
        self.glyph.data_source.data = self.get_cds_dict(
            df_filtered, psets_vertical_ordering_df, col_names, curr_selection
        )
        self.glyph_segment.data_source.data = self.get_cds_dict_segment(
            df_filtered, psets_vertical_ordering_df, col_names, curr_selection
        )
        self.glyph_wedge.data_source.data = self.get_cds_dict_wedge(
            df_filtered, psets_vertical_ordering_df, col_names, curr_selection
        )
        timer_obj.done()

    def generate_figure(self, nja_params):
        p = figure(
            width=nja_params["widthspx_nxn"],
            height=nja_params["heightspx_nxn"],
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True,
        )
        p.toolbar.logo = None
        p.min_border = 0
        p.xaxis.major_label_text_font_size = "0pt"  # turn off x-axis tick labels
        p.yaxis.major_label_text_font_size = "0pt"  # turn off y-axis tick labels
        return p


class nxn_partitions:
    def __init__(self, nja_params, col_names, dissimilarity_np):
        self.p = self.generate_figure(nja_params)
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
            nja_params,
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

    def update_selection(self, nja_params, col_names, dissimilarity_np):
        timer_obj = timer("Updating NxN Partitions")
        self.glyph.data_source.data = self.get_cds_dict(dissimilarity_np, col_names)
        self.glyph_line.data_source.data = self.get_cds_dict2(
            dissimilarity_np, col_names
        )
        timer_obj.done()

    def generate_figure(self, nja_params):
        p = figure(
            width=nja_params["widthspx_nxn"],
            height=nja_params["heightspx_nxn"],
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True,
        )
        p.toolbar.logo = None
        p.min_border = 0
        return p
