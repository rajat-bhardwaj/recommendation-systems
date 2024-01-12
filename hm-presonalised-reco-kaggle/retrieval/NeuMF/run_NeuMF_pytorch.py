import sys

sys.setrecursionlimit(5000)
sys.path.append("../")

import gc
import json
from pathlib import Path
from time import time

import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import NeuMF_pytorch_helper as neucf_models
from NeuMF_pytorch import DataModuleClass

import warnings

warnings.filterwarnings("ignore")


def run_model(neg_strategy, model_name):
    path_tensor = neucf_models.path_tensor
    path_model_ = neucf_models.path_model_

    path_model_data = Path(neucf_models.path_processed_data)
    path_base = Path(neucf_models.path_data)

    # default logger used by trainer (if tensorboard is installed)
    logger = TensorBoardLogger(
        save_dir=path_tensor,
        name="lightning_logs",
        version="neg_{}_{}_{}".format(neg_strategy, model_name, int(time())),
    )

    n_users, n_items = neucf_models.get_cust_item_count(path_base)

    with open("NeuMF_fixed_params.json") as file:
        f_params = json.load(file)

    eval_metric = f_params["eval_metric"]
    patience = f_params["patience"]

    if model_name == "gmf":
        lr = f_params["gmf_lr"]
        weight_decay = f_params["gmf_weight_decay"]
        batch_size = f_params["gmf_batch_size"]
        emb_dim_GMF = f_params["emb_dim_GMF"]
        max_epochs = f_params["gmf_max_epochs"]
        step_size = f_params["gmf_step_size"]
        gamma = f_params["gmf_gamma"]

    elif model_name == "mlp":
        lr = f_params["mlp_lr"]
        batch_size = f_params["mlp_batch_size"]
        emb_dim_MLP = f_params["emb_dim_MLP"]
        max_epochs = f_params["mlp_max_epochs"]
        weight_decay = f_params["mlp_weight_decay"]
        layers_MLP = f_params["layers_MLP"]
        step_size = f_params["mlp_step_size"]
        gamma = f_params["mlp_gamma"]
        batchnorm = f_params["batchnorm"]
        dropout = f_params["dropout"]
        dropout_rate = f_params["dropout_rate"]

    elif model_name == "neumf":
        batch_size = f_params["neumf_batch_size"]
        max_epochs = f_params["neumf_max_epochs"]
        lr = f_params["neumf_lr"]
        gamma = f_params["neumf_gamma"]
        weight_decay = f_params["neumf_weight_decay"]
        step_size = f_params["neumf_step_size"]
        pre_train = f_params["pre_train"]
        # fixed prams from pre-trained
        layers_MLP = f_params["layers_MLP"]
        emb_dim_GMF = f_params["emb_dim_GMF"]
        emb_dim_MLP = f_params["emb_dim_MLP"]
        batchnorm = f_params["batchnorm"]
        dropout = f_params["dropout"]
        dropout_rate = f_params["dropout_rate"]

    dl_param_ = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": f_params["num_workers"],
        "pin_memory": f_params["pin_memory"],
        "prefetch_factor": f_params["prefetch_factor"],
    }

    trainer_params = {
        "max_epochs": max_epochs,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": f_params["accelerator"],
        "logger": logger,
        "strategy": f_params["strategy"],
        "precision": f_params["precision"],
        ## test code
        # "limit_train_batches": f_params["limit_train_batches"],
        # disable validation
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
    }

    input_dl = DataModuleClass(
        path_model_data,
        neg_strategy,
        f_params["sample"],
        f_params["sample_frac_tr"],
        f_params["subsample_neg"],
        f_params["_num_neg"],
        dl_param_,
    )

    if model_name == "gmf":
        model_ = neucf_models.get_mf(
            dm=input_dl,
            n_users=n_users,
            n_items=n_items,
            emb_dim_GMF=emb_dim_GMF,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            step_size=step_size,
            params_=trainer_params,
            patience=patience,
            hps=False,
            eval_metric=eval_metric,
        )

    elif model_name == "mlp":
        model_ = neucf_models.get_mlp(
            dm=input_dl,
            n_users=n_users,
            n_items=n_items,
            emb_dim_MLP=emb_dim_MLP,
            layers_MLP=layers_MLP,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            gamma=gamma,
            step_size=step_size,
            params_=trainer_params,
            patience=patience,
            hps=False,
            eval_metric=eval_metric,
        )
    elif model_name == "neumf":
        if pre_train:
            pre_train_gmf = neucf_models.get_pretrained(
                f_params, "gmf", neg_strategy, n_users, n_items
            )
            pre_train_mlp = neucf_models.get_pretrained(
                f_params, "mlp", neg_strategy, n_users, n_items
            )
        else:
            pre_train_gmf = None
            pre_train_mlp = None

        model_ = neucf_models.get_neucf(
            dm=input_dl,
            n_users=n_users,
            n_items=n_items,
            layers=layers_MLP,
            dim_GMF=emb_dim_GMF,
            dim_MLP=emb_dim_MLP,
            pre_trained_GMF=pre_train_gmf,
            pre_trained_MLP=pre_train_mlp,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            gamma=gamma,
            step_size=step_size,
            params_=trainer_params,
            patience=patience,
            pre_train=pre_train,
            hps=False,
            eval_metric=eval_metric,
        )

    torch.save(
        model_.state_dict(),
        Path(path_model_)
        / "model_{}_neg_{}_{}.pt".format(model_name, neg_strategy, int(time())),
    )

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # run_model(1, "neumf")
    for sample in [1, 2, 3]:
        run_model(sample, "neumf")
