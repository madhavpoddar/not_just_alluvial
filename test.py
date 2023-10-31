# import numpy as np

# from bokeh.io import curdoc, show
# from bokeh.models import ColumnDataSource, Grid, LinearAxis, MultiLine, Plot
# from bokeh.plotting import figure

# N = 9
# x = np.linspace(-2, 2, N)
# y = x**2

# xpts = np.array([-0.09, -0.12, 0.0, 0.12, 0.09])
# ypts = np.array([-0.1, 0.02, 0.1, 0.02, -0.1])

# source = ColumnDataSource(
#     dict(
#         xs=[xpts * (1 + i / 10.0) + xx for i, xx in enumerate(x)],
#         ys=[ypts * (1 + i / 10.0) + yy for i, yy in enumerate(y)],
#     )
# )

# plot = figure(title=None, width=1500, height=700, min_border=0, toolbar_location=None)

# # glyph = MultiLine(
# #     xs="xs", ys="ys", line_color="#8073ac", line_width=2.5
# # )
# # plot.add_glyph(source, glyph)

# glyph = plot.multi_line(
#     xs=[[0, 1]], ys=[[0, 0]], line_color="#8073ac", line_width=100.5
# )
# # plot.add_glyph(glyph)

# ys = plot.y_scale
# # xaxis = LinearAxis()
# # plot.add_layout(xaxis, "below")

# # yaxis = LinearAxis()
# # plot.add_layout(yaxis, "left")

# # plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
# # plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

# # curdoc().add_root(plot)

# show(plot)


# import numpy as np
# import scipy.special

# from bokeh.layouts import gridplot
# from bokeh.plotting import figure, show


# def make_plot(title, hist, edges, x, pdf, cdf):
#     p = figure(title=title, tools="", background_fill_color="#fafafa")
#     p.quad(
#         top=hist,
#         bottom=0,
#         left=edges[:-1],
#         right=edges[1:],
#         fill_color="navy",
#         line_color="white",
#         alpha=0.5,
#     )
#     p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
#     p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

#     p.y_range.start = 0
#     p.legend.location = "center_right"
#     p.legend.background_fill_color = "#fefefe"
#     p.xaxis.axis_label = "x"
#     p.yaxis.axis_label = "Pr(x)"
#     p.grid.grid_line_color = "white"
#     return p


# # Normal Distribution

# mu, sigma = 0, 0.5

# measured = np.random.normal(mu, sigma, 1000)
# hist, edges = np.histogram(measured, density=True, bins=50)

# x = np.linspace(-2, 2, 1000)
# pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
# cdf = (1 + scipy.special.erf((x - mu) / np.sqrt(2 * sigma**2))) / 2

# p1 = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

# # Log-Normal Distribution

# mu, sigma = 0, 0.5

# measured = np.random.lognormal(mu, sigma, 1000)
# hist, edges = np.histogram(measured, density=True, bins=50)

# x = np.linspace(0.0001, 8.0, 1000)
# pdf = (
#     1
#     / (x * sigma * np.sqrt(2 * np.pi))
#     * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
# )
# cdf = (1 + scipy.special.erf((np.log(x) - mu) / (np.sqrt(2) * sigma))) / 2

# p2 = make_plot("Log Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

# # Gamma Distribution

# k, theta = 7.5, 1.0

# measured = np.random.gamma(k, theta, 1000)
# hist, edges = np.histogram(measured, density=True, bins=50)

# x = np.linspace(0.0001, 20.0, 1000)
# pdf = x ** (k - 1) * np.exp(-x / theta) / (theta**k * scipy.special.gamma(k))
# cdf = scipy.special.gammainc(k, x / theta)

# p3 = make_plot("Gamma Distribution (k=7.5, θ=1)", hist, edges, x, pdf, cdf)

# # Weibull Distribution

# lam, k = 1, 1.25
# measured = lam * (-np.log(np.random.uniform(0, 1, 1000))) ** (1 / k)
# hist, edges = np.histogram(measured, density=False, bins=50)
# print(hist)
# print(edges)


# x = np.linspace(0.0001, 8, 1000)
# pdf = (k / lam) * (x / lam) ** (k - 1) * np.exp(-((x / lam) ** k))
# cdf = 1 - np.exp(-((x / lam) ** k))

# p4 = make_plot("Weibull Distribution (λ=1, k=1.25)", hist, edges, x, pdf, cdf)

# show(gridplot([p1, p2, p3, p4], ncols=2, width=400, height=400, toolbar_location=None))

##############################################################################################
## TEA
##############################################################################################

# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_moons

# from bokeh.plotting import figure, show
# from bokeh.layouts import row

# # Generate the two moons dataset
# total_points = 10000
# dog_points = 7000
# cat_points = 3000

# # Simulate classifier predictions based on a straight-line decision boundary
# np.random.seed(42)  # For reproducibility

# # Generate the moon data for "dog" points
# moon_data_dog, _ = make_moons(n_samples=dog_points * 2, noise=0.1, shuffle=False)
# moon_data_dog = moon_data_dog[:dog_points, :]
# # Generate the moon data for "cat" points
# moon_data_cat, _ = make_moons(n_samples=cat_points * 2, noise=0.1, shuffle=False)
# moon_data_cat = moon_data_cat[cat_points:, :]

# # Create a DataFrame for "dog" and "cat" data
# data = {
#     "x": np.concatenate((moon_data_dog[:, 0], moon_data_cat[:, 0])),
#     "y": np.concatenate((moon_data_dog[:, 1], moon_data_cat[:, 1])),
#     "label": ["dog"] * dog_points + ["cat"] * cat_points,
# }
# df = pd.DataFrame(data)


# # Slope and intercept for classifier A's decision boundary
# slope_A = 1.38
# intercept_A = 0.0

# # Slope and intercept for classifier B's decision boundary
# slope_B = 0.0
# intercept_B = 0.519


# # Calculate predicted labels based on the decision boundaries
# def predict_labels(slope, intercept, data):
#     line = slope * data["x"] + intercept
#     predicted_labels = np.where(data["y"] > line, "dog", "cat")
#     return predicted_labels


# # Simulate classifier A's predicted labels
# classifierA_predicted = predict_labels(slope_A, intercept_A, df)

# # Simulate classifier B's predicted labels
# classifierB_predicted = predict_labels(slope_B, intercept_B, df)

# # Add classifier predicted labels to the DataFrame
# df["classifierA_predicted_label"] = classifierA_predicted
# df["classifierB_predicted_label"] = classifierB_predicted

# # Calculate accuracy of classifierA_predicted_label
# correct_predictions = (df["label"] == df["classifierA_predicted_label"]).sum()
# total_predictions = len(df)
# accuracy = correct_predictions / total_predictions
# print(f"Accuracy of classifier A: {accuracy:.2%}")

# correct_predictions = (df["label"] == df["classifierB_predicted_label"]).sum()
# total_predictions = len(df)
# accuracy = correct_predictions / total_predictions
# print(f"Accuracy of classifier B: {accuracy:.2%}")

# # Print the first few rows of the DataFrame
# print(df)

# # df.to_csv("Synthetic_2_classifiers.csv", index=False)


# # Create a Bokeh figure
# df["fill_color"] = "white"  # Both incorrect
# df.loc[
#     (df["label"] == df["classifierB_predicted_label"])
#     | (df["label"] == df["classifierA_predicted_label"]),
#     "fill_color",
# ] = "gray"  # One correct (+ Both correct)
# df.loc[
#     (df["label"] == df["classifierB_predicted_label"])
#     & (df["label"] == df["classifierA_predicted_label"]),
#     "fill_color",
# ] = "black"  # Both correct

# df.loc[df["label"] == "dog", "line_color"] = "red"
# df.loc[df["label"] == "cat", "line_color"] = "blue"

# p_scatter = figure(
#     width=800, height=600, title="Projected Data Space", tools=["lasso_select"]
# )

# p_scatter.scatter(
#     x="x",
#     y="y",
#     size=5,
#     source=df,
#     fill_color="fill_color",
#     line_color="line_color",
# )

# p_bar = figure(
#     width=800,
#     height=600,
#     title="Assessing Classifiers' performance for each class.",
#     tools=[],
#     y_axis_label="Count",
# )

# df_dog = df[df["label"] == "dog"]
# df_cat = df[df["label"] == "cat"]


# p_bar.quad(
#     bottom=[0, 0],
#     top=[len(df_dog), len(df_cat)],
#     left=[-0.25, 0.75],
#     right=[0.25, 1.25],
#     fill_color=["red", "blue"],
#     line_color=["red", "blue"],
#     fill_alpha=0.2,
# )

# p_bar.quad(
#     bottom=[
#         0,
#         len(df_dog[df_dog["fill_color"] == "black"]),
#         0,
#         len(df_dog[df_dog["fill_color"] == "black"]),
#         0,
#         len(df_cat[df_cat["fill_color"] == "black"]),
#         0,
#         len(df_cat[df_cat["fill_color"] == "black"]),
#     ],
#     top=[
#         len(df_dog[df_dog["fill_color"] == "black"]),
#         (df_dog["label"] == df_dog["classifierA_predicted_label"]).sum(),
#         len(df_dog[df_dog["fill_color"] == "black"]),
#         (df_dog["label"] == df_dog["classifierB_predicted_label"]).sum(),
#         len(df_cat[df_cat["fill_color"] == "black"]),
#         (df_cat["label"] == df_cat["classifierA_predicted_label"]).sum(),
#         len(df_cat[df_cat["fill_color"] == "black"]),
#         (df_cat["label"] == df_cat["classifierB_predicted_label"]).sum(),
#     ],
#     left=[-0.20, -0.20, 0.04, 0.04, 0.8, 0.8, 1.04, 1.04],
#     right=[-0.04, -0.04, 0.2, 0.2, 0.96, 0.96, 1.2, 1.2],
#     fill_color=["black", "gray", "black", "gray", "black", "gray", "black", "gray"],
#     line_color=None,
# )


# # Customize x and y axis ranges
# p_bar.y_range.start = 0  # Set y-axis range start to 0
# p_bar.x_range.start = -0.5  # Set x-axis range start to -0.5
# p_bar.x_range.end = 1.5  # Set x-axis range end to 0.5
# # Customize x-axis ticks and labels
# custom_ticks = [-0.12, 0.12, 0.88, 1.12]
# custom_tick_labels = [
#     "classifier A -\nCorrectly\nPredicted\nDog",
#     "classifier B -\nCorrectly\nPredicted\nDog",
#     "classifier A -\nCorrectly\nPredicted\nCat",
#     "classifier B -\nCorrectly\nPredicted\nCat",
# ]
# p_bar.xaxis.ticker = custom_ticks
# p_bar.xaxis.major_label_overrides = {
#     tick: label for tick, label in zip(custom_ticks, custom_tick_labels)
# }


# # Show the plot
# show(row([p_scatter, p_bar]))


import numpy as np
from sklearn.manifold import MDS
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256

# Create a matrix of dissimilarity scores (not necessarily symmetric)
# Replace this with your actual dissimilarity scores
dissimilarity_matrix = np.random.rand(100, 100)

# Make the dissimilarity matrix symmetric
symmetric_dissimilarity_matrix = (dissimilarity_matrix + dissimilarity_matrix.T) / 2.0

# Apply MDS to reduce the dimensionality to 2D
mds = MDS(n_components=1, dissimilarity="precomputed")
embedded = mds.fit_transform(symmetric_dissimilarity_matrix)

# Create a color map
mapper = linear_cmap(
    field_name="color",
    palette=Viridis256,
    low=min(embedded[:, 0]),
    high=max(embedded[:, 0]),
)

# Create a Bokeh figure
p = figure(width=800, height=800, title="MDS Visualization with Color Map")

# Add the scatter plot
p.scatter(
    x=embedded[:, 0], y=embedded[:, 0], size=10, color=mapper, legend_field="color"
)

# Add a color bar
color_bar = ColorBar(color_mapper=mapper["transform"], width=8, location=(0, 0))
p.add_layout(color_bar, "right")

# Show the plot
output_notebook()
show(p)
