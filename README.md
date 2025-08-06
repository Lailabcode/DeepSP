# DeepSP
DeepSP is an antibody-specific surrogate model that can generate 30 spatial properties of an antibody solely based on their sequence.

# How to generate descriptors (features) using DeepSP

## Option 1 - Google colab notebook
- Run
1. Prepare your input file according to the format DeepSP_input.csv
2. Run the notebook file DeepSP_predictor.ipynb
3. DeepSP structural properties for sequences inputed, will be populated and saved to a csv file - 'DeepSP_descriptor.csv'.

## Option 2 - Linux environment 
- git clone https://github.com/Lailabcode/DeepSP.git
- cd DeepSP
- Set up (bash)- create an environment and install necessary package
1. conda create -n deepSP python=3.9.13
2. source activate deepSP
3. conda install -c bioconda anarci
4. pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4
- Run
1. Prepare your input file according to the format DeepSP_input.csv
2. Run the python file deepsp_predictor.py - 'python deepsp_predictor.py'
3. DeepSP structural properties for sequences inputed, will be obtained and saved to a csv file - 'DeepSP_descriptor.csv'.


# Citation

Kalejaye, L.; Wu, I.-E.; Terry, T.; Lai, P.-K. DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability. *Comput. Struct. Biotechnol. J.* 2024, 23, 2220–2229 (https://doi.org/10.1016/j.csbj.2024.05.029)
