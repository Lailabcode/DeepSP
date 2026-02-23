# DeepSP
**DeepSP** is an antibody-specific surrogate model that can generate 30 spatial properties of an antibody solely based on their sequence.

# How to generate descriptors (features) using DeepSP

---

## Pipeline Workflow

### 1️⃣ Feature Preparation

Prepare a CSV file named:

```
DeepSP_input.csv
```

This file must contain the variable region sequences of the mAbs to be analyzed.

---

### 2️⃣ Generate Spatial Properties (DeepSP)

Run:

```
DeepSP_predictor.ipynb
```

DeepSP generates **30 spatial descriptors** from antibody sequences.

Output:

```
DeepSP_descriptors_anarci2_Abdev.csv
```

---

## 🔄 Update: Migration from ANARCI to ANARCII

AbDev has transitioned from **ANARCI** to **ANARCII** for antibody sequence numbering.

Install via:

```bash
pip install anarcii
```

### Why this change?
- pip installable
- Improved compatibility with modern Python environments  
- Simplified installation (no legacy HMMER dependency)  
- Active maintenance  

### Important Note

Due to differences in numbering logic and backend implementation, minor variations in IMGT residue assignments may occur.

## Citation

> Kalejaye, L., Wu, I.E., Terry, T., & Lai, P.K.  
> *DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability*  
> Computational and Structural Biotechnology Journal, 23:2220–2229, 2024.  
> https://www.csbj.org/article/S2001-0370(24)00173-9/fulltext  
