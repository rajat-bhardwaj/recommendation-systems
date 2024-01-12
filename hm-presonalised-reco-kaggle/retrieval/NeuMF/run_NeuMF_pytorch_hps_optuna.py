import sys

sys.setrecursionlimit(5000)
sys.path.append("../")

import gc
import json
from pathlib import Path
from time import time

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import callbacks as callbacks
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import NeuMF_pytorch_helper as neucf_models
from NeuMF_pytorch import DataModuleClass

import warnings

warnings.filterwarnings("ignore")

path_processed_data = neucf_models.path_processed_data
path_data = neucf_models.path_data
path_tensor = neucf_models.path_tensor
path_model_ = neucf_models.path_model_


def optuna_objective(trial, neg_strategy, model_name):
    """
    neg_strategy,
    model_name: "neumf", "gmf" or "mlp"
    """

    path_model_data = Path(path_processed_data)
    path_base = Path(path_data)

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

    ## setup params and hyperparams
    if model_name == "gmf":
        config = get_params_gmf(trial)
        lr = config["gmf_lr"]
        weight_decay = config["gmf_weight_decay"]
        batch_size = config["gmf_batch_size"]
        emb_dim_GMF = config["emb_dim_GMF"]
        max_epochs = config["gmf_max_epochs"]
        step_size = f_params["gmf_step_size"]
        gamma = f_params["gmf_gamma"]

    elif model_name == "mlp":
        config = get_params_mlp(trial)
        lr = config["mlp_lr"]
        batch_size = config["mlp_batch_size"]
        emb_dim_MLP = config["emb_dim_MLP"]
        max_epochs = config["mlp_max_epochs"]
        weight_decay = config["mlp_weight_decay"]
        layers_MLP = f_params["layers_MLP"]
        step_size = f_params["mlp_step_size"]
        gamma = f_params["mlp_gamma"]
        batchnorm = f_params["batchnorm"]
        dropout = f_params["dropout"]
        dropout_rate = f_params["dropout_rate"]

    elif model_name == "neumf":
        config = get_params_neumf(trial)
        batch_size = config["neumf_batch_size"]
        max_epochs = config["neumf_max_epochs"]
        lr = config["neumf_lr"]
        gamma = f_params["neumf_gamma"]
        weight_decay = f_params["neumf_weight_decay"]
        pre_train = f_params["pre_train"]
        # fixed prams from pre-trained
        layers_MLP = f_params["layers_MLP"]
        emb_dim_GMF = f_params["emb_dim_GMF"]
        emb_dim_MLP = f_params["emb_dim_MLP"]
        batchnorm = f_params["batchnorm"]
        dropout = f_params["dropout"]
        dropout_rate = f_params["dropout_rate"]
        step_size = f_params["neumf_step_size"]

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
        "enable_model_summary": False,
        "accelerator": f_params["accelerator"],
        "logger": logger,
        "strategy": f_params["strategy"],
        "precision": f_params["precision"],
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
        value, model_ = neucf_models.get_mf(
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
            eval_metric=eval_metric,
            hps=True,
        )
    elif model_name == "mlp":
        value, model_ = neucf_models.get_mlp(
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
            hps=True,
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

        value, model_ = neucf_models.get_neucf(
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
            hps=True,
        )

    torch.cuda.empty_cache()
    gc.collect()

    return value


def get_params_gmf(trial):
    """ """
    search_space = {
        "gmf_lr": trial.suggest_float("gmf_lr", 1e-5, 0.1, log=True),
        "gmf_batch_size": trial.suggest_categorical("gmf_batch_size", [512, 1024]),
        "emb_dim_GMF": trial.suggest_categorical("emb_dim_GMF", [8, 16, 32]),
        # "gmf_max_epochs": trial.suggest_int("gmf_max_epochs", 5, 20, step=5),
        "gmf_weight_decay": trial.suggest_float("gmf_weight_decay", 1e-5, 2, log=False),
        # "gmf_gamma": trial.suggest_float("gmf_gamma", 0.1, 0.9, log=False),
    }

    return search_space


def get_params_mlp(trial):
    """ """
    search_space = {
        "mlp_lr": trial.suggest_float("mlp_lr", 1e-3, 1, log=True),
        "mlp_batch_size": trial.suggest_categorical("mlp_batch_size", [32, 64, 256]),
        "emb_dim_MLP": trial.suggest_categorical("emb_dim_MLP", [128, 512]),
        "mlp_max_epochs": trial.suggest_int("mlp_max_epochs", 5, 20, step=5),
        "mlp_weight_decay": trial.suggest_float("mlp_weight_decay", 1e-3, 1, log=False),
    }

    return search_space


def get_params_neumf(trial):
    """ """
    search_space = {
        "neumf_lr": trial.suggest_float("neumf_lr", 1e-5, 0.1, log=True),
        "neumf_batch_size": trial.suggest_categorical(
            "neumf_batch_size", [4096, 8192, 16384]
        ),
        "neumf_max_epochs": trial.suggest_int("neumf_max_epochs", 5, 20, step=5),
        "neumf_weight_decay": trial.suggest_int("weight_decay", 2, 10, step=2),
        "neumf_gamma": trial.suggest_float("gamma", 0.1, 0.6, log=False),
    }

    return search_space


def run_hps_tuning(neg_strategy, n_trials, model_name):
    """ """
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="minimize",
        study_name="NeuMF_neg_{}_{}_{}".format(neg_strategy, model_name, int(time())),
        pruner=pruner,
    )

    func = lambda trial: optuna_objective(trial, neg_strategy, model_name)

    study.optimize(
        func,
        n_trials=n_trials,  # number of trials per process
        gc_after_trial=True,
        n_jobs=1,  # number of processes
        show_progress_bar=True,
    )

    # Create a dataframe from the study.
    df = study.trials_dataframe()

    filepath = "{}/lbs_neumf_optuna_neg_{}_{}_{}.csv".format(
        Path(path_data) / "optuna_results", neg_strategy, model_name, int(time())
    )
    df.to_csv(filepath)


if __name__ == "__main__":
    start_time = time()
    # run_hps_tuning(1, 1, "neumf")
    n_trials = 20

    # for neg_strategy in [1]:
    # for model_name in ["gmf"]:
    run_hps_tuning(1, n_trials, "gmf")

    print("total evaluation time ", time() - start_time)
