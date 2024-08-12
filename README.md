To run this, you need to have conda. If not, please install it from its official website. 

Once conda is installed go to the main project folder and type the following commands:

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



