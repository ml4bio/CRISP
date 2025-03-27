
from pathlib import Path
from typing import List

import pandas as pd
import torch


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_df=None,
    device="cuda",
):
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    if isinstance(data_df, str):
        df = pd.read_parquet(data_df)
    else:
        df = data_df

    if df is not None:
        emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)
    else:
        assert embedding_model == "zeros"
        emb = torch.zeros((len(smiles), 256))
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
