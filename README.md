# DeepSP

DeepSP는 항체의 서열만을 기반으로 30가지 공간 속성을 생성할 수 있는 항체 특이적 대리 모델입니다.

# DeepSP를 사용하여 디스크립터(특징)를 생성하는 방법

## 옵션 1 - Google Colab 노트북

- 실행
1. DeepSP_input.csv 형식에 따라 입력 파일을 준비합니다.
2. 노트북 파일 DeepSP_predictor.ipynb를 실행합니다.
3. 입력된 서열에 대한 DeepSP 구조적 속성이 채워지고 'DeepSP_descriptor.csv'라는 csv 파일로 저장됩니다.

## 옵션 2 - Linux 환경

- 설정 (bash) - 환경을 생성하고 필요한 패키지를 설치합니다.
1. `conda create -n deepSP python=3.9.13`
2. `source activate deepSP`
3. `conda install -c bioconda anarci`
4. `pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4`
- 실행
1. DeepSP_input.csv 형식에 따라 입력 파일을 준비합니다.
2. 파이썬 파일 deepsp_predictor.py를 실행합니다 - '`python deepsp_predictor.py`'
3. 입력된 서열에 대한 DeepSP 구조적 속성이 얻어지고 'DeepSP_descriptor.csv'라는 csv 파일로 저장됩니다.

# 인용

Kalejaye, L.; Wu, I.-E.; Terry, T.; Lai, P.-K. DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability. Comput. Struct. Biotechnol. J. 2024, 23, 2220–2229 (https://doi.org/10.1016/j.csbj.2024.05.029)