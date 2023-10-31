from os.path import dirname, join, basename
import pandas as pd
import time

import pickle


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

