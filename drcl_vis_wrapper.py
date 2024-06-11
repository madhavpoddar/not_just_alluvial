from bokeh.layouts import row, column
from bokeh.models import Tabs, TabPanel
from bokeh.events import Tap, DoubleTap
from bokeh.plotting import figure
import copy
import os
import pickle

import pandas as pd

import hashlib

from df_preprocessing import (
    reduce_intersections_neighbours,
    calc_ARI_matrix,
)
from draw_vis import (
    alluvial,
    cim,
    # ndimplot,
    mds_col_similarity_cl_membership,
    mds_nxn_setwise,
    # similarity_roof_shaped_matrix_diagram,
)
from vis_interaction import (
    calc_curr_selection,
    selection_update_tap,
    set_initial_curr_selection,
    df_assign_colors,
)
from vis_params import set_skewer_params
from helper_functions_project_specific import column_names
from color_mapping_sets import get_setwise_color_allocation


def is_number(s):
    try:
        # Try to convert the string to a float
        number = float(s)
        return number  # Return the float if it's not an integer
    except ValueError:
        return None  # Return None if the conversion fails


def generate_dataframe_hash(df):
    # Hash the DataFrame and convert to a string representation
    hashed_values = pd.util.hash_pandas_object(df).values

    # Convert hashed values to bytes
    byte_representation = hashed_values.tobytes()

    # Generate a hash code using hashlib
    hash_code = hashlib.md5(byte_representation).hexdigest()

    return hash_code


def is_filename_in_subfolder(filename, subfolder="preprocessing"):
    # Get the current directory
    current_dir = os.getcwd()

    # Construct the path to the subfolder
    subfolder_path = os.path.join(current_dir, subfolder)

    # Check if the subfolder exists
    if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
        # Check if the filename exists in the subfolder
        filename_path = os.path.join(subfolder_path, filename)
        if os.path.exists(filename_path) and os.path.isfile(filename_path):
            return True
        else:
            return False
    else:
        return False


def pickle_objects(obj1, obj2, filename, subfolder="preprocessing"):
    # Get the current directory
    current_dir = os.getcwd()

    # Construct the path to the subfolder
    subfolder_path = os.path.join(current_dir, subfolder)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Construct the full path to the file
    file_path = os.path.join(subfolder_path, filename)

    # Pickle the objects and write to the file
    with open(file_path, "wb") as f:
        pickle.dump((obj1, obj2), f)

    print(f"Preprocessing objects have been pickled and stored in '{file_path}'.")


def unpickle_objects(pickle_filename, subfolder="preprocessing"):
    # Get the current directory
    current_dir = os.getcwd()

    # Construct the path to the subfolder
    subfolder_path = os.path.join(current_dir, subfolder)

    # Check if the subfolder exists
    if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
        print(f"Error: Sub-folder '{subfolder}' does not exist.")
        return None, None

    # Construct the full path to the file
    file_path = os.path.join(subfolder_path, pickle_filename)

    # Check if the file exists
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(
            f"Error: File '{pickle_filename}' not found in the '{subfolder}' sub-folder."
        )
        return None, None

    # Unpickle the objects from the file
    with open(file_path, "rb") as f:
        obj1, obj2 = pickle.load(f)

    print(f"The objects have been unpickled from '{file_path}'.")
    return obj1, obj2


class drcl_vis_wrapper:
    def __init__(
        self,
        df,
        col_names,
        col_names_as_list_of_numbers=None,
        sequential_variable_name=None,
        bool_reduce_intersections_neighbours=False,
    ):
        if sequential_variable_name == None:
            sequential_variable_name = "Sequential Variable"
        if col_names_as_list_of_numbers == None:
            col_names_are_categorical = False
            col_names_as_list_of_numbers = []
            for col_name in col_names:
                col_name_number = is_number(col_name)
                # If there is even one non-numeric column name,
                # all will be treated like categorical
                if col_name_number == None:
                    col_names_are_categorical = True
                    break
                else:
                    col_names_as_list_of_numbers.append(col_name_number)

            if col_names_are_categorical:
                col_names_as_list_of_numbers = list(range(len(col_names)))

        elif len(col_names_as_list_of_numbers) != len(col_names):
            print("len(col_names_as_list_of_numbers) != len(col_names). Terminating...")
            exit(1)
        column_details_df = pd.DataFrame(
            {sequential_variable_name: col_names_as_list_of_numbers}, index=col_names
        )
        self.col_names = column_names(col_names)

        if bool_reduce_intersections_neighbours:
            self.df = reduce_intersections_neighbours(df, self.col_names)
        else:
            self.df = df

        self.skewer_params = set_skewer_params(self.df, self.col_names)

        if len(column_details_df.columns) > 1:
            print("Currently only one level of Sequence is suppoerted.")
            exit(1)

        column_details_df[self.skewer_params["random_tag"]] = range(
            len(column_details_df.index)
        )
        column_details_df = column_details_df[
            [self.skewer_params["random_tag"]]
            + [
                col
                for col in column_details_df.columns
                if col != self.skewer_params["random_tag"]
            ]
        ]
        self.curr_selection = set_initial_curr_selection(self.col_names)

        pickle_filename = generate_dataframe_hash(self.df[self.col_names]) + ".pkl"
        if is_filename_in_subfolder(pickle_filename):
            self.psets_color, self.psets_vertical_ordering_df = unpickle_objects(
                pickle_filename
            )
        else:
            (
                self.psets_color,
                self.psets_vertical_ordering_df,
            ) = get_setwise_color_allocation(self.df[self.col_names])
            pickle_objects(
                self.psets_color, self.psets_vertical_ordering_df, pickle_filename
            )

        df_assign_colors(
            self.df,
            self.psets_color,
            self.curr_selection["color_col_name"],
            self.skewer_params["color_col_name"],
            remove_colors=len(self.curr_selection["cluster_ids"]) == 0
            and self.curr_selection["color_col_name"] == None,
        )
        self.df_filtered = self.df
        self.fig_obj = {}
        self.fig_obj["alluvial"] = alluvial(
            self.df,
            column_details_df,
            self.psets_vertical_ordering_df,
            self.psets_color,
            self.skewer_params,
            self.col_names,
            self.curr_selection,
        )
        self.fig_obj["cim"] = cim(
            self.fig_obj["alluvial"].p.x_range,
            self.df,
            self.skewer_params,
            self.psets_color,
            self.col_names,
            self.curr_selection,
        )
        # self.fig_obj["metamap_edit_dist"] = metamap_edit_dist_pt_grps(
        #     self.df,
        #     self.skewer_params,
        #     self.col_names,
        #     self.curr_selection,
        # )
        # self.fig_obj["ndimplot"] = ndimplot(
        #     self.df,
        #     self.skewer_params,
        #     self.col_names,
        #     self.curr_selection,
        # )
        self.fig_obj["mds_col_similarity_cl_membership"] = (
            mds_col_similarity_cl_membership(
                self.skewer_params,
                self.col_names,
                1 - calc_ARI_matrix(self.df, self.col_names),
            )
        )
        self.fig_obj["mds_nxn_setwise"] = mds_nxn_setwise(
            self.df,
            self.skewer_params,
            self.col_names,
            self.curr_selection,
            self.psets_vertical_ordering_df,
        )
        # self.fig_obj[
        #     "similarity_roof_shaped_matrix_diagram"
        # ] = similarity_roof_shaped_matrix_diagram(
        #     self.skewer_params,
        #     self.col_names,
        #     1 - calc_ARI_matrix(self.df, self.col_names),
        # )

        self.fig_obj["alluvial"].rbg_edge_alpha_highlight.on_change(
            "active", self.rbg_alluvial_edge_alpha_highlight_handler
        )
        self.fig_obj["alluvial"].p.on_event(Tap, self.tap_callback)
        self.fig_obj["alluvial"].p.on_event(DoubleTap, self.tap_callback)
        # self.fig_obj["ndimplot"].multichoice_cols.on_change(
        #     "value", self.ndimplot_multichoice_cols_handler
        # )

        # TODO: remove this line later
        # self.empty_fig0 = figure(width=800, height=800)
        # self.empty_fig1 = figure(width=300, height=390)

        self.layout = self.generate_layout()

    def generate_layout(self):
        # l00a = column(
        #     children=[
        #         self.fig_obj["ndimplot"].p,
        #         self.fig_obj["ndimplot"].multichoice_cols,
        #     ]
        # )
        # l00b = column(
        #     children=[
        #         self.fig_obj["metamap_edit_dist"].p1,
        #         self.fig_obj["metamap_edit_dist"].p2,
        #     ]
        # )
        l01a = column(
            children=[
                # self.fig_obj["alluvial"].rbg_edge_alpha_highlight,
                self.fig_obj["alluvial"].p,
                self.fig_obj["cim"].p_normal,
                self.fig_obj["cim"].p_inverted,
                # self.fig_obj["cim"].cim_setwise_details_or_not_cbgrp,
            ]
        )

        # l01b = self.empty_fig0
        # l01c = row(
        #     children=[
        #         column(
        #             children=[
        #                 self.fig_obj["mds_col_similarity_cl_membership"].p,
        #                 self.fig_obj["similarity_roof_shaped_matrix_diagram"].p,
        #             ]
        #         ),
        #         self.fig_obj["similarity_roof_shaped_matrix_diagram"].data_table,
        #     ]
        # )
        l01c = self.fig_obj["mds_col_similarity_cl_membership"].p
        # l02 = self.empty_fig1

        # Interative Clustering
        # data_space_view_panel = TabPanel(child=l00a, title="Data Space View")
        # l00b_panel = TabPanel(child=l00b, title="Edit Distance based Analysis")
        # Columns (Dis-)similarity based on Cluster-Membership
        # panel_1x1x1_1xn = TabPanel(child=l01a, title="Sequential Comparison")
        # l01b_panel = TabPanel(child=l01b, title="1xN Comparison")
        panel_nxn_partition = TabPanel(child=l01c, title="NxN Comparison (Partion)")
        panel_nxn_set = TabPanel(
            child=self.fig_obj["mds_nxn_setwise"].p, title="NxN Comparison (Set)"
        )
        # Cluster Positioning (Estimated??)
        # l02a_panel = TabPanel(child=l02, title="Cluster Positions*")
        # l02b_panel = TabPanel(child=l02, title="Pt. Trails*")
        # Selected Label Colours Interaction
        # l12a_panel = TabPanel(child=l02, title="Labels Overlapping")

        # tabs_1x1x1_1xn = Tabs(tabs=[panel_1x1x1_1xn])
        tabs_nxn_partition = Tabs(tabs=[panel_nxn_partition])
        tabs_nxn_set = Tabs(tabs=[panel_nxn_set])
        # tabs_l10 = Tabs(tabs=[l10b_panel])
        # tabs_l02 = Tabs(tabs=[l02a_panel, l02b_panel])
        # tabs_l12 = Tabs(tabs=[l12a_panel])

        final_layout = row(
            children=[
                l01a,
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
            self.skewer_params,
        )

    # def ndimplot_multichoice_cols_handler(self, attr, old, new):
    #     self.curr_selection["ndimplot_col_names"] = [x for x in new]
    #     self.fig_obj["ndimplot"].update_selection(
    #         self.df_filtered,
    #         self.skewer_params,
    #         self.col_names,
    #         self.curr_selection,
    #         self.curr_selection,
    #     )

    def rbg_alluvial_edge_alpha_highlight_handler(self, attr, old, new):
        self.fig_obj["alluvial"].alluvial_edges_obj.rbg_edge_alpha_highlight_active = (
            new
        )
        self.fig_obj["alluvial"].alluvial_edges_obj.update_selection(
            self.df,
            self.df_filtered,
            self.psets_vertical_ordering_df,
            self.psets_color,
            self.skewer_params,
            self.col_names,
            self.curr_selection,
        )
