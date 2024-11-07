# DeepSP
DeepSP is an antibody-specific surrogate model that can generate 30 spatial properties of an antibody solely based on their sequence.

# How to generate descriptors (features) using DeepSP (can be run on google colab notebook OR on a linux environment).

## Google colab notebook
- Prepare your input file according to the format DeepSP_input.csv
- Run the notebook file DeepSP_predictor.ipynb
- DeepSP structural properties for sequences inputed, will be populated and saved to a csv file - 'DeepSP_descriptor.csv'.

## Linux environment 
- create an environment and install necessary package
	conda create -n deepSP python=3.9.13
	source activate deepSP
	conda install -c bioconda anarci
	pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4

- Prepare your input file according to the format DeepSP_input.csv
- Run the python file deepsp_predictor.py - 'python deepsp_predictor.py'
- DeepSP structural properties for sequences inputed, will be obtained and saved to a csv file - 'DeepSP_descriptor.csv'.


# Citation

Kalejaye, L.; Wu, I.-E.; Terry, T.; Lai, P.-K. DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability. *Comput. Struct. Biotechnol. J.* 2024, 23, 2220–2229 (https://doi.org/10.1016/j.csbj.2024.05.029)