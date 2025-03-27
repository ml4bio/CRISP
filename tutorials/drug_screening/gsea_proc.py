import scanpy as sc 
import pandas as pd 
import numpy as np 
import gseapy as gp
import pickle
import CRISP.trainer as ct
import CRISP.data as cd

def stratified_sample_adata(adata, obs_column, fraction=None, samp_num=None,random_state=42):
    np.random.seed(random_state)
    sampled_indices = []

    unique_groups = adata.obs[obs_column].unique()
    size=samp_num if samp_num != None else int(len(group_indices) * fraction)

    for group in unique_groups:
        group_indices = adata.obs[adata.obs[obs_column] == group].index
        sampled_group_indices = np.random.choice(
            group_indices, 
            size=size, 
            replace=False, 
        )
        sampled_indices.extend(sampled_group_indices)

    sampled_adata = adata[sampled_indices].copy()

    return sampled_adata

def calculate_accuracy(a, b):
    a_set, b_set = set(a), set(b)
    correct_predictions = len(a_set & b_set)  
    accuracy = correct_predictions / len(a_set) if a_set else 0 
    recall = correct_predictions / len(b_set) if b_set else 0
    return accuracy, recall

def main():

    # path of perturbation dataset (.h5ad)
    datapath = '/PATH/TO/sciplex_pp_hvgenes_scFM_resplit.h5ad'
    # key of cell type
    cov_name = 'cell_type'
    # key of split setting
    split='split2'
    # target ood cell type
    cov = 'K562'
    # dosage
    dose = 1.0

    adata = sc.read(datapath)
    adata.var['gene_name'] = adata.var_names.copy()

    # Load trained model
    exp = ct.Trainer()
    model_path = f'/PATH/TO/model.pt'
    exp.load_model(model_path)
    _, _, drugs_name_to_idx =cd.drug_to_idx(np.array(adata.obs['condition'].values))
    drugs = adata[adata.obs[split]=='ood'].obs['condition'].unique()
    
    acc_dict = {}

    # Predict for each drug
    for drug in drugs:
        adata_treated = adata[(adata.obs['condition']==drug)&(adata.obs[split]=='ood')&(adata.obs[cov_name]==cov)&(adata.obs['dose_val']==dose)]
        adata_ctrl = adata[(adata.obs['control']==1)&(adata.obs[split]=='ood')&(adata.obs[cov_name]==cov)]

        if len(adata_treated) <= 5:
            print(f'drug {drug} has no ground truth')
            continue

        # Subset dataset if too large
        if len(adata_ctrl) > 1000:
            adata_ctrl=stratified_sample_adata(adata_ctrl,obs_column=cov_name,samp_num=1000,random_state=42)
        if len(adata_treated) > 1000:
            adata_treated = stratified_sample_adata(adata_treated,obs_column=cov_name,samp_num=1000,random_state=42)

        # Prediction
        adata_pred, _, _ = exp.get_prediction(adata_ctrl,drug_name=drug,dose=dose,ref_drug_dict=drugs_name_to_idx)

        pred = adata_pred.X.copy()
        ctrl = adata_ctrl.X.A.copy()
        true = adata_treated.X.A.copy()
        gene_names = adata.var['gene_name'].values.copy()

        pred_df = pd.DataFrame(pred.T,index=gene_names)
        pred_df.columns = ['Perturb'] * pred_df.shape[1]
        ctrl_df = pd.DataFrame(ctrl.T, index=gene_names)
        ctrl_df.columns = ['Ctrl'] * ctrl_df.shape[1]
        true_df = pd.DataFrame(true.T, index=gene_names)
        true_df.columns = ['True'] * true_df.shape[1]

        # perform GSEA analysis of predicted results
        df = pd.concat([pred_df,ctrl_df],axis=1)
        df.index = df.index.astype(str)
        gs = gp.gsea(data=df, 
                    gene_sets='c2.cp.v2024.1.Hs.symbols.gmt',# or 'h.all.v2024.1.Hs.symbols.gmt'
                    cls = list(df.columns),
                    min_size=10,
                    permutation_type='phenotype',
                    permutation_num=1000,
                    outdir=None,
                    method='signal_to_noise',
                    threads=4,
                    seed=42)

        # perform GSEA analysis of ground truth
        df = pd.concat([true_df,ctrl_df],axis=1)
        df.index = df.index.astype(str)
        gs_true = gp.gsea(data=df, 
                    gene_sets='c2.cp.v2024.1.Hs.symbols.gmt',# or 'h.all.v2024.1.Hs.symbols.gmt'
                    cls = list(df.columns), 
                    min_size=10,
                    permutation_type='phenotype',
                    permutation_num=1000, 
                    outdir=None,
                    method='signal_to_noise',
                    threads=4,
                    seed=42)
        
        # Filter significant terms
        gseadf_t = gs_true.res2d.copy()
        gseadf_neg_t = gseadf_t[(gseadf_t.NES<-1)&(gseadf_t['FDR q-val']<0.25)]
        gseadf_pos_t = gseadf_t[(gseadf_t.NES>1)&(gseadf_t['FDR q-val']<0.25)]
        gseadf = gs.res2d.copy()
        gseadf_neg = gseadf[(gseadf.NES<-1)&(gseadf['FDR q-val']<0.05)]
        gseadf_pos = gseadf[(gseadf.NES>1)&(gseadf['FDR q-val']<0.05)]

        if (len(gseadf_neg_t)<1) and (len(gseadf_pos_t)<1):
            print(f"drug {drug} enrichment results are not significant")
            continue 

        # Save GSEA results
        pd.concat([gseadf_pos_t,gseadf_neg_t]).to_csv(f'./gsea/{drug}_gsea_truth.csv')
        pd.concat([gseadf_pos,gseadf_neg]).to_csv(f'./gsea/{drug}_gsea_pred.csv')

        true_neg_term = gseadf_neg_t['Term']
        pred_neg_term = gseadf_neg['Term']
        true_pos_term = gseadf_pos_t['Term']
        pred_pos_term = gseadf_pos['Term']

        # calculate precisioin and recall
        pre_neg,recall_neg = calculate_accuracy(pred_neg_term,true_neg_term)
        pre_pos,recall_pos = calculate_accuracy(pred_pos_term,true_pos_term)

        print(f'Drug: {drug}')
        print(f'Len of neg: {len(gseadf_neg_t)}, Len of pos: {len(gseadf_pos_t)}')
        print(f'Pos accuracy: {pre_pos}, Pos recall: {recall_pos}')
        print(f'Neg accuracy: {pre_neg}, Neg recall: {recall_neg}')

        acc_dict[drug] = {'neg':len(gseadf_neg_t),'pos':len(gseadf_pos_t),'pos_acc':pre_pos,'pos_recall':recall_pos,'neg_acc':pre_neg,'neg_recall':recall_neg}
    
    # Save result
    with open(f'./{cov}_gsea_acc.pkl','wb') as f:
        pickle.dump(acc_dict,f)


if __name__ == '__main__':
    main()
