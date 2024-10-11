# DeepSP
DeepSP is an antibody-specific surrogate model that can generate 30 spatial properties of an antibody solely based on their sequence.

# How to generate descriptors (features) using DeepSP

DeepSP can be run either using an IPython notebook or as a callable script.

### Using IPython notebook
- Prepare your input file in the same format as DeepSP_input.csv
- Run the notebook file DeepSP_predictor.ipynb
- DeepSP structural properties for input sequences would be calculated and saved
to `DeepSP_descriptors.csv`.

### As a callable script
- Activate conda environment from `environment.yml`
- Prepare your input either as a CSV or as a directory of FASTAs (each should
contain one antibody, with heavy and light chain IDs postfixed `_VH` and `_VL`
respectively)
- Call `./DeepSP_predict.py -i <input files> --in_format <fasta|csv> -o <out.csv>`

# Citation

Kalejaye, L.; Wu, I.-E.; Terry, T.; Lai, P.-K. DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability. *Comput. Struct. Biotechnol. J.* 2024, 23, 2220–2229 (https://doi.org/10.1016/j.csbj.2024.05.029)


