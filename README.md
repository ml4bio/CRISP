# CRISP
This repository contains code for the paper **Predicting Drug Responses of Unseen Cell Types through Transfer Learning with Foundation Models**.

## Installation
First, we need to create a conda environment
```bash
conda create -n crisp_env python=3.9  
conda activate crisp_env
```

Follow the code below to install CRISP. Installation may take about 2 minute. Notably, if you are using CUDA, make sure that the version of PyTorch you install is compatible with your CUDA environment. You should install the PyTorch version that matches your CUDA setup.
```bash
git clone https://github.com/ml4bio/CRISP.git
cd CRISP
pip install -r requirement.txt
pip install -e .
```

## Quick Start

Training: \
Follow the [tutorial notebook for training](/tutorials/training.ipynb). Each training may take 30-60 minutes depends on size of dataset. Or you can directly train with script below. Files of configs and shell scripts are provided in [experiments/](experiments/) for replication of results. 

```bash
python CRISP/train_script.py --config [path/to/config.yaml] --split split --seed 0 --savedir [path/to/save/folder]
```

Prediction with trained model: \
Follow the [tutorial notebook](/tutorials/zeroshot_prediction.ipynb). We provide the trained [model parameter](https://drive.google.com/drive/folders/1QWjmpYZMaqxfLwIeLjwoz-H9vX60udeu?usp=drive_link) from Neurips for running prediction.

## Data

Preprocessed datasets used in this work all can be downloaded [here](https://drive.google.com/drive/folders/1QWjmpYZMaqxfLwIeLjwoz-H9vX60udeu?usp=drive_link). There are four perturbation datasets: NeurIPS, SciPlex3, GBM, PC9, and one normal dataset PBMC-Bench. The code of data preprocessing is provided in [data folder](data/)






