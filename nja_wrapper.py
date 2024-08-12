import copy
from bokeh.layouts import row, column
from bokeh.models import Tabs, TabPanel
from bokeh.events import Tap, DoubleTap

from nja_params import set_nja_params
from nja_preprocessing import (
    generate_dataframe_hash,
    is_filename_in_subfolder,
    pickle_objects,
    unpickle_objects,
    calc_ARI_matrix,
    calc_sets_colors_2d_positions,
)
from nja_vis_encodings import (
    alluvial,
    ixn_merge_split,
    nxn_partitions,
    nxn_sets,
)
from nja_interactions import (
    calc_curr_selection,
    selection_update_tap,
    set_initial_curr_selection,
    df_assign_colors,
)
from nja_helper_functions import column_names


class nja_wrapper:
    def __init__(
        self,
        df,
        col_names,
        bool_pickle_preprocessing=True,
    ):
        self.col_names = column_names(col_names)
        self.df = df
        self.df_filtered = self.df
        self.nja_params = set_nja_params(self.df, self.col_names)
        self.curr_selection = set_initial_curr_selection(self.col_names)

        #####################################################################################
        ## Pre-processing steps
        #####################################################################################
        if bool_pickle_preprocessing:
            pickle_filename = generate_dataframe_hash(self.df[self.col_names]) + ".pkl"
            if is_filename_in_subfolder(pickle_filename):
                self.psets_color, self.psets_vertical_ordering_df = unpickle_objects(
                    pickle_filename
                )
            else:
                (
                    self.psets_color,
                    self.psets_vertical_ordering_df,
                ) = calc_sets_colors_2d_positions(self.df[self.col_names])
                pickle_objects(
                    self.psets_color, self.psets_vertical_ordering_df, pickle_filename
                )
        else:
            (
                self.psets_color,
                self.psets_vertical_ordering_df,
            ) = calc_sets_colors_2d_positions(self.df[self.col_names])
        df_assign_colors(
            self.df,
            self.psets_color,
            self.curr_selection["color_col_name"],
            self.nja_params["color_col_name"],
            remove_colors=len(self.curr_selection["cluster_ids"]) == 0
            and self.curr_selection["color_col_name"] == None,
        )

        #####################################################################################
        ## Generating the visualizations
        #####################################################################################
        self.fig_obj = {}
        self.fig_obj["alluvial"] = alluvial(
            self.df,
            self.psets_vertical_ordering_df,
            self.psets_color,
            self.nja_params,
            self.col_names,
            self.curr_selection,
        )
        self.fig_obj["ixn_merge_split"] = ixn_merge_split(
            self.fig_obj["alluvial"].p.x_range,
            self.df,
            self.nja_params,
            self.psets_color,
            self.col_names,
            self.curr_selection,
        )
        self.fig_obj["nxn_partitions"] = nxn_partitions(
            self.nja_params,
            self.col_names,
            1 - calc_ARI_matrix(self.df, self.col_names),
        )
        self.fig_obj["nxn_sets"] = nxn_sets(
            self.df,
            self.nja_params,
            self.col_names,
            self.curr_selection,
            self.psets_vertical_ordering_df,
        )

        #####################################################################################
        ## Adding listners for click/double-click events
        #####################################################################################
        self.fig_obj["alluvial"].p.on_event(Tap, self.tap_callback)
        self.fig_obj["alluvial"].p.on_event(DoubleTap, self.tap_callback)

        #####################################################################################
        ## Arranging the visualizations in a specific layout
        #####################################################################################
        self.layout = self.generate_layout()

    def generate_layout(self):
        panel_nxn_partition = TabPanel(
            child=self.fig_obj["nxn_partitions"].p,
            title="NxN Comparison (Partition)",
        )
        panel_nxn_set = TabPanel(
            child=self.fig_obj["nxn_sets"].p, title="NxN Comparison (Set)"
        )
        tabs_nxn_partition = Tabs(tabs=[panel_nxn_partition])
        tabs_nxn_set = Tabs(tabs=[panel_nxn_set])
        final_layout = row(
            children=[
                column(
                    children=[
                        self.fig_obj["alluvial"].p,
                        self.fig_obj["ixn_merge_split"].p_normal,
                        self.fig_obj["ixn_merge_split"].p_inverted,
                    ]
                ),
                column(children=[tabs_nxn_partition, tabs_nxn_set]),
            ],
            spacing=20,
        )
        return final_layout

    def tap_callback(self, event):
        old_selection = copy.deepcopy(self.curr_selection)
        self.curr_selection = calc_curr_selection(
            event, old_selection, self.psets_vertical_ordering_df, self.col_names
        )
        print("Current selection: " + str(self.curr_selection))
        self.df, self.df_filtered = selection_update_tap(
            self.curr_selection,
            old_selection,
            self.df,
            self.fig_obj,
            self.col_names,
            self.psets_vertical_ordering_df,
            self.psets_color,
            self.nja_params,
        )
