# CRISP
CRISP is developed for predicting drug perturbation response for unseen cell types in cell type-specific way. It incorporates single cell foundation models (scFMs) into a cell type-specific learning framework. By exploiting cell type similarities and divergences learned from FMs, CRISP effectively extends existing cell atlases into the perturbation space, providing a systematic approach to characterize drug-induced cellular state transitions. During the inference stage, CRISP requires only drug information and control state scRNA-seq data as input, to predict the cell type-specific drug responses.

## Installation
Follow the code below to install CRISP. Installation may take about 1 minute.
```bash
git clone https://github.com/Susanxuan/CRISP.git
cd CRISP
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






