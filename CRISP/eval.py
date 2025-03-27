import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics import R2Score

from CRISP.data import SubDataset
from CRISP.model import PertAE
from CRISP.losses import sinkhorn_dist,energy_dist,gaussian_mmd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse


def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    """
    return np.where(x)[0]


def repeat_n(x, n):
    """
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def mean(x: list):
    """
    Returns mean of list `x`
    """
    return np.mean(x) if len(x) else -1


def compute_r2(y_true, y_pred):
    """
    Computes the r2 score for `y_true` and `y_pred`,
    returns `-1` when `y_pred` contains nan values
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    metric = R2Score()
    metric.update(y_pred, y_true) 
    return metric.compute().item()

def compute_cor(y_true, y_pred):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    cor_mtx = torch.corrcoef(torch.stack([y_true,y_pred],dim=0))
    return cor_mtx[0,1].item()

def compute_prediction_CRISP(autoencoder, genes, cell_embeddings, emb_drugs, emb_covs=None, drugs_pre=None):

    gene_pred, latent_treated, mu, logvar = autoencoder.predict(
            genes=genes,
            cell_embeddings=cell_embeddings,
            drugs_idx=emb_drugs[0],
            dosages=emb_drugs[1],
            covariates=emb_covs,
            drugs_pre=drugs_pre,
        )
    gene_pred = gene_pred.detach()
    latent_treated = latent_treated.detach()
    mu = mu.detach()
    
    return gene_pred, latent_treated, mu

def evaluate(autoencoder: PertAE, treated_dataset: SubDataset, control_dataset: SubDataset):
    """
    Conduct evaluation using pearson correlation coefficiency and r2 coefficiency, MSE on both DE genes and All genes.
    It's performed for each perturbation group by condition like 'celltype_drug_dose'. 
    The metrics are calculated from the mean of predicted perturbed profile and true perturbed profile.
    
    Outputs: 
        (1). Dict of overall evaluation results. 
        (2). Dict of evaluation results for each perturbation group
        (3). Dict of prediction results for each perturbation group
    """
    eval_score_dict = {}
    pred_dict = {}

    genes_control = control_dataset.genes
    genes_true = treated_dataset.genes

    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(treated_dataset.pert_categories, dtype="category")
    de_genes = treated_dataset.de_genes
    var_names = treated_dataset.var_names

    for cell_drug_dose_comb, category_count in zip(
        *np.unique(treated_dataset.pert_categories, return_counts=True)
    ):

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.

        if len(list(de_genes.keys())[0].split('_')) == 2:
            cell_drug_comb = cell_drug_dose_comb.split('_')[0]+'_'+cell_drug_dose_comb.split('_')[1]
            bool_de = var_names.isin(
                np.array(de_genes[cell_drug_comb])
            )
        else:
            bool_de = var_names.isin(
                np.array(de_genes[cell_drug_dose_comb])
            )
        idx_de = bool2idx(bool_de) # the var index of de genes

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb) # bool list showing where is the members of this cell_drug_dose_comb group
        idx_all = bool2idx(bool_category) # the obs index of this cell_drug_dose_comb group
        idx = idx_all[0]

        ct = cell_drug_dose_comb.split('_')[0]
        genes_control_sub = genes_control[control_dataset.celltype == ct].to('cuda')
        if len(genes_control_sub) < 5:
            continue
        n_rows = genes_control_sub.size(0)

        if treated_dataset.covariates is not None:
            emb_covs = [repeat_n(cov[idx], n_rows) for cov in treated_dataset.covariates] # n_rows is the number of control obs
        else:
            emb_covs = None
        emb_drugs = (
            repeat_n(treated_dataset.drugs_idx[idx], n_rows).squeeze(),
            repeat_n(treated_dataset.dosages[idx], n_rows).squeeze(),
        )

        cell_embeddings_sub = control_dataset.paired_cell_embeddings[control_dataset.celltype == ct].to('cuda')

        preds = compute_prediction_CRISP(
            autoencoder,
            genes_control_sub,
            cell_embeddings_sub,
            emb_drugs,
            emb_covs,
        )[0]

        y_true = genes_true[idx_all, :]
        preds = preds.detach().to('cpu')

        ctrl_m = genes_control_sub.mean(dim=0).to('cpu')
        yt_m = y_true.mean(dim=0).to('cpu')
        yp_m = preds.mean(dim=0).to('cpu')

        metrics_dict=calc_metrics(yt_m, yp_m, ctrl_m, y_true, preds, idx_de)
        eval_score_dict[cell_drug_dose_comb] = metrics_dict
        pred_dict[cell_drug_dose_comb] = {'true':yt_m,'pred':yp_m,'ctrl':ctrl_m}

    metrics_dict_all = {}
    for k,v in eval_score_dict.items():
        for k_, v_ in v.items():
            if k_ in list(metrics_dict_all.keys()):
                metrics_dict_all[k_] += [v_]
            else:
                metrics_dict_all[k_] = [v_]

    for k,v in metrics_dict_all.items():
        metrics_dict_all[k] = np.mean(v)
    
    return metrics_dict_all, eval_score_dict, pred_dict


def calc_metrics(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
    metrics_dict = {}
    yt_de_m = yt_m[idx_de]
    yp_de_m = yp_m[idx_de]
    if yt_de_m.sum() == 0:
        yt_de_m[0] = yt_de_m[0] + 1e-6
    if yp_de_m.sum() == 0:
        yp_de_m[0] = yp_de_m[0] + 1e-6

    metrics_dict['r2score'] = max(compute_r2(yt_m, yp_m),0)
    metrics_dict['r2score_de'] = max(compute_r2(yt_m[idx_de], yp_m[idx_de]),0)
    metrics_dict['pearson'] = pearsonr(yt_m, yp_m)[0]
    metrics_dict['pearson_de'] = pearsonr(yt_de_m, yp_de_m)[0]
    metrics_dict['mse'] = mse(yt_m,yp_m)
    metrics_dict['mse_de'] = mse(yt_m[idx_de],yp_m[idx_de])
    metrics_dict['pearson_delta'] = pearsonr(yt_m-ctrl_m,yp_m-ctrl_m)[0]
    metrics_dict['pearson_delta_de'] = pearsonr(yt_m[idx_de]-ctrl_m[idx_de],yp_m[idx_de]-ctrl_m[idx_de])[0]

    # metrics_dict['mmd'] = gaussian_mmd(y_true,preds).item()
    # metrics_dict['mmd'] = 0
    # metrics_dict['mmd_de'] = gaussian_mmd(y_true[:,idx_de],preds[:,idx_de]).item()

    # if (preds.sum()==0) & (y_true.sum()==0):
    #     metrics_dict['sinkhorn'] = 0
    # else:
    #     metrics_dict['sinkhorn'] = 0

    if (preds[:,idx_de].sum()==0) & (y_true[:,idx_de].sum()==0):
        metrics_dict['sinkhorn_de'] = 0
    else:
        metrics_dict['sinkhorn_de'] = sinkhorn_dist(y_true[:,idx_de],preds[:,idx_de]).item()
        
    # metrics_dict['energy'] = energy_dist(y_true,preds).item()
    # metrics_dict['energy'] = 0
    # metrics_dict['energy_de'] = energy_dist(y_true[:,idx_de],preds[:,idx_de]).item()

    # deal with nan value
    if np.isnan(metrics_dict['pearson']):
        metrics_dict['pearson'] = 0
    if np.isnan(metrics_dict['pearson_de']):
        metrics_dict['pearson_de'] = 0
    if np.isnan(metrics_dict['pearson_delta']):
        metrics_dict['pearson_delta'] = 0
    if np.isnan(metrics_dict['pearson_delta_de']):
        metrics_dict['pearson_delta_de'] = 0

    return metrics_dict