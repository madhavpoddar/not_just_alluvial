To run this, you need conda. 

Once conda is installed go to the main project folder and type the following commands:
<code>

conda env create --name nja --file=nja_env.yml

conda activate nja

bokeh serve --show hd_cluster_analysis.py

</code>