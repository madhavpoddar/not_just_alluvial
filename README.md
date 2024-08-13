This multi-view visual analytics system is used analyze changes in partition sequences. It is an prototype implementation of an approach described in the following research paper:

Not Just Alluvial: Towards a More Comprehensive Visual Analysis of Data Partition Sequences. Madhav Poddar, Jan-Tobias Sohns, and Fabian Beck. Vision, Modeling, and Visualization 2024.

The Not Just Alluvial (NJA) prototype was developed using the [Bokeh library](https://bokeh.pydata.org/en/latest/) in Python.

# Running the tool

Prerequisites: conda needs to be installed.

To run the application simply clone the repository, go to the main project folder and start with the following commands:

<code>

conda env create --name nja --file=nja_env.yml

conda activate nja

</code>

This will create the conda environment and activate it.

Next, to run the code: 

<code>

bokeh serve --show nja.py

</code>

If you want to visualize a different sample dataset or provide a different dataset, please make modifications in the file "nja.py". If you provide your own dataset and it is large (with respect to number of sets), please run the following two commands instead (to avoid server timeout issues):

<code>

python nja.py

bokeh serve --show nja.py

</code>

# Project structure

| Source File Name         | Description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| nja.py                   | The main file where you specify the dataset to visualize and start the bokeh server.           |
| nja_sample_df.py         | Used for loading/creating sample datasets.                                                     |
| nja_wrapper.py           | Wrapper class (called by nja.py) that connects the project source files.                       |
| nja_preprocessing.py     | Contains functions executed during the pre-processing steps.                                   |
| nja_vis_encodings.py     | Defines how the data is visualized in the 4 component views.                                   |
| nja_interactions.py      | Interprets the interactions events and initiates the update of the component views.            |
| nja_params.py            | Defines the parameters associated with each component (e.g. height of the alluvial diagram).   |
| nja_helper_funtions.py   | Contains helper functions.                                                                     |


| Directory Name           | Description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| data                     | contains the sample dataset files.                                                             |
| preprocessing            | contains the pre-processing files (unique with respect to input dataset).                      | 

