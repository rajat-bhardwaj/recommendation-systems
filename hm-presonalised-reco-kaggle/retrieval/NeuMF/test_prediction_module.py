import sys

sys.path.append("../")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch import callbacks as callbacks
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import model_summary
from torch.utils.data import DataLoader, TensorDataset

from NeuMF_pytorch import DataModuleClass
from NeuMF_pytorch import GeneralisedMatrixFactorisation as gmf
from NeuMF_pytorch import MultiLayerPerceptron as mlp
from NeuMF_pytorch import MyCallback
from NeuMF_pytorch import NeuMatrixFactorisation as ncf


class PredictionDataModule(pl.LightningDataModule):
    def __init__(self, dataloader_params):
        super().__init__()
        self.dataloader_params = dataloader_params

    def prepare_data(self):
        observed_truth = pd.read_parquet(
            ""
        )
        avail_products = pd.read_parquet(
            ""
        )

        if True:
            observed_truth = observed_truth.sample(frac=0.1)
            avail_products = avail_products.sample(frac=0.1)

        avail_products = avail_products.article_id.unique()

        self.avail_custs = torch.tensor(observed_truth.customer_id.unique()).reshape(
            -1, 1
        )
        self.avail_products = torch.tensor(avail_products).reshape(-1, 1)

        self.n_custs = self.avail_custs.shape[0]
        self.n_prods = self.avail_products.shape[0]

        self.custs = self.avail_custs.repeat_interleave(self.n_prods)
        self.prod = self.avail_products.repeat(self.n_custs, 1)

    def setup(self, stage):
        # each customer will be repeated n_prods times
        self.dataset = TensorDataset(self.custs, self.prod)

        # generate dataset by cross joining customers with products in train set
        # run hit rate and NDCG as metric
        # NDCG is not muc of a use as there is no original ranking.
        # the order observed during the aggregation process is arbitrary

    # in __getitem__: we can construct a dataframe for each customer

    def predict_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_params)

    def __len__(self):
        return len(self.dataset)


def get_cust_item_count(path_base):
    """ """
    cust = pd.read_csv(path_base / "lookup_table_customers.csv")
    prod = pd.read_csv(path_base / "lookup_table_products.csv")

    n_users = cust["_customer_id"].nunique()
    n_items = prod["_article_id"].nunique()

    return n_users, n_items


def run_prediction():
    path_processed_data = Path(
        ""
    )
    path_data = Path("")

    n_users, n_items = get_cust_item_count(path_data)

    dl_param_ = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 5,  # subprocesses to use for data loading
        "pin_memory": False,  # TRUE for GPU
        "prefetch_factor": 50,
    }

    trainer_params = {
        "max_epochs": 2,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "mps",
        "devices": 1,  # number of devices use for training
        "default_root_dir": "./",
        "precision": "16-mixed",
        "limit_train_batches": 50,
        "limit_val_batches": 50,
    }

    input_dataloader = DataModuleClass(
        path_processed_data, 1, True, 0.01, 0.1, True, 5, dl_param_
    )

    callbacks_ = MyCallback()
    callbacks_ = callbacks_.get_callbacks()

    model_gmf = gmf(
        n_users=n_users,
        n_items=n_items,
        dim=16,
        lr=0.001,
        weight_decay=0,
        gamma=0.1,
        step_size=10,
    )

    trainer_gmf = pl.Trainer(callbacks=callbacks_, **trainer_params)
    trainer_gmf.fit(model=model_gmf, datamodule=input_dataloader)

    predict_dl_config = {
        "batch_size": 10000,
        "drop_last": False,
        "shuffle": False,
        "num_workers": 5,  # subprocesses to use for data loading
        "pin_memory": False,  # TRUE for GPU
        "prefetch_factor": 100,
    }

    ref = PredictionDataModule(predict_dl_config)
    results = trainer_gmf.predict(
        model=model_gmf, datamodule=ref, return_predictions=True
    )


if __name__ == "__main__":
    run_prediction()
