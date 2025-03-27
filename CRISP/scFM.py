from scgpt.tasks import embed_data
import scanpy as sc

def calc_gpt(adata,model_path,gene_name='gene_name',return_key='X_scGPT'):
    adata_add = embed_data(adata,model_path,gene_name,return_new_adata=True)
    adata.obsm[return_key] = adata_add.X.copy()

    return adata