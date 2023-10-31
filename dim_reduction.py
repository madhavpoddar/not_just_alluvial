from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_DR_data(df):
    print("Performing Dimensionality Reduction")
    pca = PCA(n_components=df.shape[1]).fit(df)
    # visualize_pca_explained_variance_ratio_(pca)
    dim_reduced_data = pca.fit_transform(df)
    scaler = StandardScaler()
    dim_reduced_data = scaler.fit_transform(dim_reduced_data)
    # visualize_pca_projected_2d(projected)
    print("Dimensionality Reduction Done")
    return dim_reduced_data
