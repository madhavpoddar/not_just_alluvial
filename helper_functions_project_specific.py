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
