from pathlib import Path

import torch
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from NeuMF_pytorch import GeneralisedMatrixFactorisation as gmf
from NeuMF_pytorch import MultiLayerPerceptron as mlp
from NeuMF_pytorch import MyCallback
from NeuMF_pytorch import NeuMatrixFactorisation as ncf

path_processed_data = ""
path_data = ""
path_tensor = ""
path_model_ = ""


def get_cust_item_count(path_base):
    """ """
    cust = pd.read_csv(path_base / "lookup_table_customers.csv")
    prod = pd.read_csv(path_base / "lookup_table_products.csv")

    n_users = cust["_customer_id"].nunique()
    n_items = prod["_article_id"].nunique()

    return n_users, n_items


def get_mf(
    dm,
    n_users,
    n_items,
    emb_dim_GMF,
    lr,
    weight_decay,
    gamma,
    step_size,
    params_,
    patience=3,
    hps=False,
    eval_metric="val_loss",
):
    callbacks_ = MyCallback(eval_metric, patience)
    callbacks_ = callbacks_.get_callbacks()

    model_gmf = gmf(
        n_users=n_users,
        n_items=n_items,
        dim=emb_dim_GMF,
        lr=lr,
        weight_decay=weight_decay,
        gamma=gamma,
        step_size=step_size,
    )

    trainer_gmf = pl.Trainer(callbacks=callbacks_, **params_)
    trainer_gmf.fit(
        model=model_gmf,
        datamodule=dm,
    )

    if hps:
        value = trainer_gmf.callback_metrics[eval_metric].item()
        return value, model_gmf
    else:
        return model_gmf


def get_mlp(
    dm,
    n_users,
    n_items,
    emb_dim_MLP,
    layers_MLP,
    lr,
    weight_decay,
    dropout,
    dropout_rate,
    batchnorm,
    gamma,
    step_size,
    params_,
    patience=3,
    hps=False,
    eval_metric="val_loss",
):
    callbacks_ = MyCallback(eval_metric, patience)
    callbacks_ = callbacks_.get_callbacks()

    model_mlp = mlp(
        n_users=n_users,
        n_items=n_items,
        dim_MLP=emb_dim_MLP,
        layers=layers_MLP,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        dropout_rate=dropout_rate,
        batchnorm=batchnorm,
        gamma=gamma,
        step_size=step_size,
    )

    trainer_mlp = pl.Trainer(callbacks=callbacks_, **params_)
    trainer_mlp.fit(model=model_mlp, datamodule=dm)

    if hps:
        value = trainer_mlp.callback_metrics[eval_metric].item()
        return value, model_mlp
    else:
        return model_mlp


def get_neucf(
    dm,
    n_users,
    n_items,
    layers,
    dim_GMF,
    dim_MLP,
    pre_trained_GMF,
    pre_trained_MLP,
    lr,
    weight_decay,
    dropout,
    dropout_rate,
    batchnorm,
    gamma,
    step_size,
    params_,
    pre_train,
    patience=3,
    hps=False,
    eval_metric="val_loss",
):
    callbacks_ = MyCallback(eval_metric, patience)
    callbacks_ = callbacks_.get_callbacks()

    model_neucf = ncf(
        n_users=n_users,
        n_items=n_items,
        layers=layers,
        dim_GMF=dim_GMF,
        dim_MLP=dim_MLP,
        pre_train=pre_train,
        pre_trained_GMF=pre_trained_GMF,
        pre_trained_MLP=pre_trained_MLP,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        dropout_rate=dropout_rate,
        batchnorm=batchnorm,
        gamma=gamma,
        step_size=step_size,
    )

    trainer_ncf = pl.Trainer(callbacks=callbacks_, **params_)
    trainer_ncf.fit(model=model_neucf, datamodule=dm)

    if hps:
        value = trainer_ncf.callback_metrics[eval_metric].item()
        return value, model_neucf
    else:
        return model_neucf


def get_pretrained(_params, model_name, neg_stg, n_users, n_items):
    files = list(Path(path_model_).rglob("*.pt"))
    for file in files:
        name_ = file.parts[-1].split("_")
        if name_[1] == model_name and int(name_[3]) == neg_stg:
            model_path = file

    if model_name == "mlp":
        model_ = mlp(
            n_users,
            n_items,
            _params["emb_dim_MLP"],
            _params["layers_MLP"],
            _params["mlp_lr"],
            _params["mlp_weight_decay"],
            _params["dropout"],
            _params["dropout_rate"],
            _params["batchnorm"],
            _params["mlp_gamma"],
            _params["mlp_step_size"],
        )
        model_.load_state_dict(torch.load(model_path))

    elif model_name == "gmf":
        model_ = gmf(
            n_users,
            n_items,
            _params["emb_dim_GMF"],
            _params["gmf_lr"],
            _params["gmf_weight_decay"],
            _params["gmf_gamma"],
            _params["gmf_step_size"],
        )
        model_.load_state_dict(torch.load(model_path))

    elif model_name == "neumf":
        model_ = ncf(
            n_users=n_users,
            n_items=n_items,
            layers=_params["layers_MLP"],
            dim_GMF=_params["emb_dim_GMF"],
            dim_MLP=_params["emb_dim_MLP"],
            pre_train=False,
            pre_trained_GMF=None,
            pre_trained_MLP=None,
            lr=_params["neumf_lr"],
            weight_decay=_params["neumf_weight_decay"],
            dropout=_params["dropout"],
            dropout_rate=_params["dropout_rate"],
            batchnorm=_params["batchnorm"],
            gamma=_params["neumf_gamma"],
            step_size=_params["neumf_step_size"],
        )
        model_.load_state_dict(torch.load(model_path))

    return model_
