"""
The Trainer framework is adapted from chemCPA
"""
import logging
import math
from typing import List, Optional, Union
import os
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
import numpy as np
import torch
import pickle
import copy
import scanpy as sc
from tqdm import tqdm

from CRISP.data import Dataset, custom_collate
from CRISP.embedding import get_chemical_representation
from CRISP.model import PertAE
from CRISP.eval import evaluate, compute_prediction_CRISP


class Trainer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def init_dataset(
        self,
        adata_obj,
        perturbation_key: Union[str, None],
        dose_key: Union[str, None],
        smiles_key: Union[str, None],
        celltype_key='cell_type',
        covariate_keys=None,
        FM_key='X_scGPT',
        control_key: str = 'control',
        pc_cov: str='cell_type',
        degs_key: str = "rank_genes_groups_cov",
        pert_category: str = "cov_drug_dose_name",
        split_ood: bool = True,
        split_key: str = "split",
        seed=0,
        use_FM=True,
        ):
        """
        adata_obj: adata path str or adata object
        perturbation_key: The column name of treatment name (like drug name).
        dose_key: The column name of treatment dosage
        smiles_key: The column name of SMILE string of drug
        celltype_key: The column name of cell type in obs dataframe
        covariate_keys: The column names of other covariates in obs, input as list format.
        FM_key: The name of FM embedding for cells in adata.obsm. 
            We extract FM embedding before training to avoid the FM's forward process in each iteration thus save time and memory
        control_key: Name of control samples in perturbation condition colunm
        pc_cov: The name of column in adata.obs to identify paired control group, 
            e.g 'celltype_donor' means find paired control group with the same celltype and donor attribute 
        degs_key: The name of DEG's for each celltype_treatment group in adata.uns
        pert_category: The name of column storing group condition for evaluation.
            e.g: cell type + drug name + drug dose. This is used during contrastive sampling and evaluation. Should align with deg dict's keys
        split_ood: if there is any ood sample in adata object
        split_key: The column name of split state (show train/test/ood state)
        use_FM: whether using FM embedding or gene expression vector
        """
        dataset = Dataset(
            adata_obj,
            perturbation_key,
            dose_key,
            celltype_key,
            covariate_keys,
            smiles_key,
            FM_key,
            degs_key,
            pert_category,
            control_key,
            split_key,
            pc_cov,
            seed,
            use_FM,
        )

        if split_ood:
            self.datasets = {
                "training": dataset.subset("train", "all"),
                "test_treated": dataset.subset("test", "treated"),
                "test_control": dataset.subset('test','control'),
                "ood_treated": dataset.subset('ood','treated'),
                "ood_control": dataset.subset('ood','control'),
            }
        else:
            self.datasets = {
                "training": dataset.subset("train", "all"),
                "test_treated": dataset.subset("test", "treated"),
                "test_control": dataset.subset('test','control'),
            }
        del dataset

    def init_drug_embedding(self, chem_model: str, chem_df):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drug_embeddings = get_chemical_representation(
            smiles=self.datasets['training'].canon_smiles_unique_sorted,
            embedding_model=chem_model,
            data_df=chem_df,
            device=device,
        )
    
    def init_model(
        self,
        hparams='',
        mmd_co=0.1,
        celltype_co=1,
        seed=0,
    ):
        self.autoencoder = PertAE(
            self.datasets["training"].num_genes,
            self.datasets["training"].num_drugs,
            self.datasets['training'].num_celltypes,
            self.datasets["training"].num_covariates,
            drug_embeddings=self.drug_embeddings,
            mmd_co=mmd_co,
            celltype_co=celltype_co,
            device=self.device,
            seed=seed,
            hparams=hparams,
            FM_ndim=self.datasets["training"].paired_cell_embeddings.shape[1],
        )

    def load_model(self,model_path):
        mo = torch.load(model_path,map_location=torch.device('cpu'))
        self.autoencoder = PertAE(**mo[1],drug_embeddings=torch.nn.Embedding.from_pretrained(mo[0]['drug_embeddings.weight']),device=self.device)
        self.autoencoder.load_state_dict(mo[0])

    def get_prediction(self, 
                       adata_ctrl, 
                       drug_name=None, 
                       dose=None, 
                       ref_drug_dict=None, 
                       FM_emb='X_scGPT', 
                       smile=None, 
                       smile_df=None, 
                       return_adata=True
                       ):
        '''
        The function of predict outcomes given control state and perturbation condition.
        For perturbation information, you should at least provide (1).drug name and ref_drug_dict, or (2).smile string and smile_df.
        
        Inputs:
            adata_ctrl: adata of control state sc matrix
            drug_name: name of drug
            ref_drug_dict: dict of {'drug_name':'drug_idx'}, and align with the trained model
            FM_emb: Column name of pretrained FM embedding in adata_ctrl.obsm
            smile: SMILE string of drug
            smile_df: dataframe containing chemical pre-embedding of each smile(drug)
            return_adata: whether return adata or tensor

        Outputs:
            1. predicted perturbation outcomes
            2. latent embeddings with perturbation 
            3. mu of infered latent control embedding from control gene expression

        '''
        
        self.autoencoder.eval()
        cell_embs = torch.tensor(adata_ctrl.obsm[FM_emb],device=self.device)
        try:
            genes = torch.tensor(adata_ctrl.X.A,device=self.device)
        except:
            genes = torch.tensor(adata_ctrl.X,device=self.device)
        
        n_rows = cell_embs.shape[0]
        if (drug_name is not None) and (drug_name in ref_drug_dict.keys()):
            emb_drugs = (
                torch.tensor([ref_drug_dict[drug_name] for i in range(n_rows)],dtype=torch.long,device=self.device),
                torch.tensor([dose for i in range(n_rows)],dtype=torch.float,device=self.device)
                )
            drugs_pre=None
        else:
            assert (smile is not None) and (smile_df is not None)
            emb_drugs = (
                None,
                torch.tensor([dose for i in range(n_rows)],dtype=torch.float,device=self.device)
            )

            drugs_pre = torch.tensor(np.array([smile_df.loc[smile].values]*n_rows), dtype=torch.float, device=self.device)

        preds, latent_treated, mu = compute_prediction_CRISP(
            self.autoencoder,
            genes,
            cell_embs,
            emb_drugs,
            emb_covs=None,
            drugs_pre=drugs_pre,
        )
        
        if return_adata:
            adata_pred = sc.AnnData(preds.cpu().numpy())
            adata_lat = sc.AnnData(latent_treated.cpu().numpy())
            adata_mu = sc.AnnData(mu.cpu().numpy())

            adata_pred.obs = adata_ctrl.obs.copy()
            adata_pred.obs['condition'] = drug_name
            adata_lat.obs = adata_pred.obs.copy()
            adata_mu.obs = adata_pred.obs.copy()

            return adata_pred, adata_lat, adata_mu
        else:
            return preds, latent_treated, mu


    def load_train(self):
        """
        Instantiates a torch DataLoader for the given batchsize
        """
        self.datasets.update(
            {
                "loader_tr": torch.utils.data.DataLoader(
                    self.datasets["training"],
                    batch_size=self.autoencoder.hparams["batch_size"],
                    collate_fn=custom_collate,
                    shuffle=True,
                    drop_last=True,
                )
            }
        )

    def train(
        self,
        num_epochs: int,
        max_minutes: int,
        checkpoint_freq: int,
        save_dir: str,
        eval_ood=True, # whether to conduct ood evaluation
    ):
        
        assert save_dir is not None
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir()

        start_time = time.time()
        for epoch in tqdm(range(num_epochs)):
            epoch_training_stats = defaultdict(float)

            for data in self.datasets["loader_tr"]:
                genes, paired_cell_embeddings, drugs_idx, dosages, degs, celltype_idx = data[:6]
                
                neg_genes, neg_paired_cell_embeddings, neg_drugs_idx, neg_dosages, neg_degs, neg_celltype_idx = data[6:12]

                covariates,neg_covariates = data[12], data[13]

                training_stats = self.autoencoder.iter_update(
                    genes=genes,
                    cell_embeddings=paired_cell_embeddings,
                    drugs_idx=drugs_idx,
                    dosages=dosages,
                    degs=degs,
                    celltype_idx=celltype_idx,
                    covariates=covariates,
                    neg_genes=neg_genes,
                    neg_cell_embeddings=neg_paired_cell_embeddings,
                    neg_drugs_idx=neg_drugs_idx,
                    neg_dosages=neg_dosages,
                    neg_degs=neg_degs,
                    neg_celltype_idx=neg_celltype_idx,
                    neg_covariates=neg_covariates,
                )

                for key, val in training_stats.items():
                    epoch_training_stats[key] += val

            self.autoencoder.scheduler_autoencoder.step()
            self.autoencoder.scheduler_cell.step()
            if self.autoencoder.num_drugs > 0:
                self.autoencoder.scheduler_dosers.step()

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if key not in self.autoencoder.history.keys():
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(val)
            self.autoencoder.history["epoch"].append(epoch)

            # print some stats for each epoch
            epoch_training_stats["epoch"] = epoch
            logging.info("\n%s", pformat(dict(epoch_training_stats), indent=4, width=1))

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history["elapsed_time_min"] = ellapsed_minutes
            reconst_loss_is_nan = math.isnan(
                epoch_training_stats["loss_reconstruction"]
            )

            stop = (
                ellapsed_minutes > max_minutes
                or (epoch == num_epochs - 1)
                or reconst_loss_is_nan
            )

            # we always run the evaluation when training has stopped
            if ((epoch % checkpoint_freq) == 0 and epoch > 0) or stop:
                evaluation_stats = {}
                evaluation_stats_all = {}
                prediction_all = {}

                with torch.no_grad():
                    self.autoencoder.eval()
                    evaluation_stats['iid'], evaluation_stats_all['iid'], prediction_all['iid'] = evaluate(
                        self.autoencoder,
                        self.datasets["test_treated"],
                        self.datasets['test_control'],
                    )
                    if eval_ood:
                        evaluation_stats['ood'], evaluation_stats_all['ood'], prediction_all['ood'] = evaluate(
                            self.autoencoder,
                            self.datasets["ood_treated"],
                            self.datasets['ood_control'],
                        )
                    
                    self.autoencoder.train()

                test_score = (
                    np.mean(list(evaluation_stats["iid"].values()))
                    if evaluation_stats["iid"]
                    else None
                )

                test_score_is_nan = test_score is not None and math.isnan(test_score)
                stop = stop or test_score_is_nan

                if stop:
                    file_name = f'model.pt'
                    torch.save(
                        (
                            self.autoencoder.state_dict(),
                            self.autoencoder.init_args,
                            self.autoencoder.history,
                        ),
                        os.path.join(save_dir, file_name),
                    )
                    logging.info(f"model_saved: {file_name}")

                    with open(save_dir+'/eval_stats.pkl','wb') as f:
                        pickle.dump(evaluation_stats,f)
                    with open(save_dir+'/eval_stats_all.pkl','wb') as f:
                        pickle.dump(evaluation_stats_all,f)  
                    with open(save_dir+'/pred_mean.pkl','wb') as f:
                        pickle.dump(prediction_all,f)
                if (
                    stop
                    and not reconst_loss_is_nan
                    and not test_score_is_nan
                ):
                    for key, val in evaluation_stats.items():
                        if key not in self.autoencoder.history:
                            self.autoencoder.history[key] = []
                        self.autoencoder.history[key].append(val)
                    self.autoencoder.history["stats_epoch"].append(epoch)

                # print some stats for the evaluation
                stats = {
                    "epoch": epoch,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                    "max_minutes_reached": ellapsed_minutes > max_minutes,
                    "max_epochs_reached": epoch == num_epochs - 1,
                }

                logging.info("\n%s", pformat(stats, indent=4, width=1))

        results = self.autoencoder.history
        results["total_epochs"] = epoch
        return results
