import warnings
from typing import Optional

# import dgl
import pandas as pd
import scanpy as sc
from rdkit import Chem
import yaml
import random

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    # config['model']['hparams']["lr"] = float(config['model']['hparams']["lr"])
    # config['model']['hparams']["wd"] = float(config['model']['hparams']["wd"])
    # config['model']['hparams']["cell_wd"] = float(config['model']['hparams']["cell_wd"])
    # config['model']['hparams']["dropout"] = float(config['model']['hparams']["dropout"])

    return config

def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    logfc_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        logfc_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['logfoldchanges'])
        # print(adata_cov.uns["rank_genes_groups"].keys())
        # break
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()
            logfc_dict[group] = logfc_genes[group].tolist()

    adata.uns[key_added] = gene_dict
    adata.uns[f'{key_added}_logfc'] = logfc_dict

    if return_dict:
        return gene_dict, logfc_dict


def canonicalize_smiles(smiles: Optional[str]):
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None

def sample_neg(adata, split_key, cov_drug_key, condition_key,seed):
    random.seed(seed)
    adata_train = adata[adata.obs[split_key]=='train']
    grouped_adata = adata_train.obs.groupby(cov_drug_key, observed=False)

    index_to_num = dict(zip(adata_train.obs.index,range(len(adata_train))))
    grouped_idx = grouped_adata.apply(lambda group: [index_to_num[i] for i in group.index])
    grouped_idx = dict(grouped_idx)

    grouped_adata = adata_train.obs.groupby(condition_key,observed=False)
    cond_idx = grouped_adata.apply(lambda group: [index_to_num[i] for i in group.index])
    cond_idx = dict(cond_idx)

    grouped_comp_idx = {}
    for k,v in grouped_idx.items():
        # ct = k.split('_')[0]
        dg = k.split('_')[1]

        grouped_comp_idx[k] = list(set(cond_idx[dg]) - set(v))

    # positive_indices = []
    negative_indices = []
    for i in adata_train.obs[cov_drug_key].values:
        # print(i)
        # positive_idx = random.choice(grouped_idx[i])
        # positive_indices.append(positive_idx)
        if len(grouped_comp_idx[i]) > 0:
            negative_idx = random.choice(grouped_comp_idx[i])
        else: 
            negative_idx = random.choice(range(len(adata_train)))
        negative_indices.append(negative_idx)

    del grouped_idx, grouped_comp_idx
    
    # pos_emb = pretrain_embs[positive_indices]
    # neg_emb = pretrain_embs[negative_indices]

    return negative_indices