from os.path import join
import pandas as pd
import time

import pickle


def get_unique_vals(df, col_name):
    return sorted(df[col_name].unique())


def get_selected_cluster_id_vals(df, col_name, selected_cluster_ids):
    cluster_id_vals = get_unique_vals(df, col_name)
    return [
        cluster_id_vals[selected_cluster_id]
        for selected_cluster_id in selected_cluster_ids
    ]


def get_cluster_count(df, col_name, c_id):
    return df.loc[df[col_name] == c_id].shape[0]


class column_names:
    def __init__(self, col_names):
        if len(list(set(col_names))) == len(col_names):
            self.col_names = col_names
        else:
            print("Error: The column names list contains duplicate elements.")

    def __len__(self):
        return len(self.col_names)

    def __iter__(self):
        return iter(self.col_names)

    def __getitem__(self, col_index: int):
        if col_index != None:
            if col_index < len(self.col_names) and col_index >= 0:
                return self.col_names[col_index]
        print("Invalid column index (" + str(col_index) + ").")

    def index(self, col_name):
        return self.col_names.index(col_name)

    def get_list(self):
        return self.col_names

    def get_neighbouring_pairs_l2r(self):
        return [
            (self.col_names[i - 1], self.col_names[i])
            for i in range(len(self.col_names) - 1, 0, -1)
        ]

    def get_col_id(self, col_name: str):
        if col_name in self.col_names:
            return self.col_names.index(col_name)
        else:
            print("Column Name " + col_name + " not found in the selected list.")


def to_empty_dict(glyph_vars):
    empty_dict = {}
    for glyph_var in glyph_vars:
        empty_dict[glyph_var] = []
    return empty_dict


class timer:
    def __init__(self, start_statement):
        self.start = time.time()
        print(start_statement, end="... ")

    def done(self):
        print("Done (" + str(round(time.time() - self.start, 2)) + "s)")


def print_df_properties(df):
    print("Index names: " + str(df.index.names))
    print("Column names: " + str(df.columns.values))
    print("Row count: " + str(len(df.index)) + "\n")
    print("Index - Count of unique values: ")
    for i in range(len(df.index.names)):
        print(df.index.names[i] + ": " + str(len(df.index.unique(level=i))))


def print_df_properties2(df, c_col_name=None):
    print("Dataframe properties: ")
    print("\n(# Instances, # Attributes) : " + str(df.shape))
    if c_col_name != None:
        print(
            "\nUnique values of the class column with indvl. counts:\n"
            + str(df[c_col_name].value_counts())
            + "\n"
        )


def df_remove_col(df, c_col_name):
    if c_col_name != None:
        df = df.loc[:, df.columns != c_col_name]
    return df


def datafile_path(filename):
    return join("data", filename)


def read_csv_file(filename, index_cols=None, display_df_properties=False):
    read_csv_se = timer("Reading " + datafile_path(filename))
    df = pd.read_csv(datafile_path(filename))
    if index_cols != None:
        df.set_index(index_cols, inplace=True)
        df = df.sort_index()
    if display_df_properties:
        print_df_properties(df)
    read_csv_se.done()
    return df


def rename_column_names(df, X_dict):
    df = df.rename(
        columns=lambda x: X_dict[x.upper()] if x.upper() in X_dict.keys() else x
    )


def save_obj(obj, obj_name):
    with open(datafile_path(obj_name + ".pckl"), "wb") as f:
        pickle.dump(obj, f)


def load_obj(obj_name):
    with open(datafile_path(obj_name + ".pckl"), "rb") as f:
        obj = pickle.load(f)
    return obj
