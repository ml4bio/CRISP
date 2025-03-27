import os
from pathlib import Path
import pprint
import argparse
import logging
import sklearn
import copy
from CRISP.utils import load_config
from CRISP.trainer import Trainer
import yaml
from pprint import pformat
import pickle
import torch
import pandas as pd

"""
The defualt usage of CRISP is: 
    python CRISP/train_script.py --config [your path of the config file] 
                                --split [the key of split type] 
                                --savedir [the path of saved model, log file, evaluation files]
                                --seed (Optional)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--split", type=str, required=True, help="Split key for data")
    parser.add_argument("--seed", type=int, required=True, default=0, help="Seed")
    parser.add_argument("--savedir", type=str, required=True, help="Path of save model")
    parser.add_argument("--MMD",type=float,default=0.1,help="coefficient of mmd loss")
    parser.add_argument("--celltype_co",type=float,default=1,help="coefficient of celltype-specific loss")
    parser.add_argument("--FM_key",type=str, default=None,help="key of FM embeddings")
    parser.add_argument("--drug_emb", type=str, default=None, help="Path of drug embeddings")
    parser.add_argument("--data_path", type=str, default=None, help="Path of adata")

    pars_args = parser.parse_args()
    config_path = pars_args.config

    args = load_config(config_path)
    args['dataset']['split_key'] = pars_args.split
    args["model"]['seed'] = pars_args.seed
    args['training']['save_dir'] = pars_args.savedir
    args['model']['mmd_co'] = pars_args.MMD
    args['model']['celltype_co'] = pars_args.celltype_co
    if pars_args.FM_key is not None:
        args['dataset']['FM_key'] = pars_args.FM_key
    if pars_args.drug_emb is not None:
        args['model']['drug_emb'] = pars_args.drug_emb
    if pars_args.data_path is not None:
        args['dataset']['adata_obj'] = pars_args.data_path
    formatted_str = pprint.pformat(args)

    log_path = args['training']['save_dir']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename= f'{log_path}/log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info(f'Argument setting: {formatted_str}')
    yaml.dump(
        args, open(f"{log_path}/config.yaml", "w"), default_flow_style=False
    )

    exp = Trainer()
    exp.init_dataset(**args["dataset"],seed=args["model"]['seed'])
    logging.info(f'Finish init dataset')
    exp.init_drug_embedding(chem_model='rdkit',chem_df=args["model"]["drug_emb"])
    logging.info(f'Finish init drug embedding')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp.init_model(
        mmd_co=args['model']['mmd_co'],
        celltype_co=args['model']['celltype_co'],
        seed=args["model"]['seed'],
    )
    exp.load_train()
    logging.info(f'Start training')
    eval_ood = args['dataset']['split_ood']
    exp.train(**args["training"],eval_ood=eval_ood)
    logging.info(f'Finish training')