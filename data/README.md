## Preparation of dataset
We have provided [preprocessed datasets](https://drive.google.com/drive/folders/1QWjmpYZMaqxfLwIeLjwoz-H9vX60udeu?usp=drive_link) to use directly.

For researchers interested in understanding the complete data preprocessing workflow, we provide detailed documentation in our nips_data.ipynb notebook. This notebook includes:

* A comprehensive explanation of the required data structure and fields
* Step-by-step code for creating an AnnData object from raw data
* Data cleaning and standardization procedures
* Implementation of the train-test-OOD split for model evaluation

The notebook provides a transparent view of how our preprocessed datasets were created, which is particularly helpful for researchers who may want to apply our methods to their own datasets or understand the underlying data structure requirements

### Training:
Before conducting training, you need to complete several preprocess stages for your datasets.

1. Prepare the raw anndata as .h5ad file or other format. It should contain both perturbed and unperturbed samples.
2. Perform normalization and log1p processing using scanpy.
3. Create columns of drug condition and cell type annotation, as well as the combination group ('drug_celltype' or 'drug_dose_celltype')
4. If nessecary, only subset top 5000 or 10000 gene features to save memory and time.
4. Identify dict of differential expression genes (DE) for each combination group. It will be used in DE-focused MSE loss calculation.
4. Prepare columns of SMILES string for each sample.
5. Create chemical embedding file as dataframe, containing the pre-embedding for each drug calculated from rdkit or other methods. NA values should be removed.
6. Calculate scFM-embeddings for each sample using scGPT or other scFMs.


### Prediction:
For predicting on new dataset with a trained CRISP model. You only need to prepare the unperturbed scRNA-seq dataset and treatment condition.

1. Prepare an unperturbed preprocessed (normalized and log1p) scRNA-seq dataset and calculate scFM-embeddings.
2. Prepare drug treatment condition. It can be a SMILES string and a dataframe containing pre-embeddings for SMILES strings, or a drug name that has shown during training. 

