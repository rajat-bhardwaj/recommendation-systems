from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import callbacks as callbacks
from torch.utils.data import DataLoader

from NeuMF_pytorch import GeneralisedMatrixFactorisation as gmf
from NeuMF_pytorch import MultiLayerPerceptron as mlp
from NeuMF_pytorch import NeuMatrixFactorisation as ncf


class MovieLensDataModule(pl.LightningDataModule):
    """ """

    def __init__(self, ds_type):  # , batch_size: int = 32):
        super().__init__()
        self.ds_type = ds_type
        # self.batch_size = batch_size

        self.custom_setup()

    def custom_setup(
        self,
    ):
        self.movie_lens = pd.read_csv(
            "../ml-100k/u.data",
            sep="\t",
            header=None,
        )
        self.movie_lens.columns = ["user", "item", "purchase", "time_"]
        self.movie_lens = self.movie_lens.filter(["user", "item", "purchase"])
        # generate negative
        self.movie_lens.purchase = np.where(self.movie_lens.purchase > 1, 0, 1)

        self.n_users = self.movie_lens.user.nunique()
        self.n_items = self.movie_lens.item.nunique()

        self.train = self.movie_lens.sample(frac=0.7)
        self.validation = self.movie_lens[~self.movie_lens.index.isin(self.train.index)]

        if self.ds_type == "train":
            self.user, self.item, self.label = self.convert_to_tensors(self.train)
        elif self.ds_type == "val":
            self.user, self.item, self.label = self.convert_to_tensors(self.validation)

    # def train_dataloader(self):
    #     return DataLoader(self.train, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.validation, batch_size=self.batch_size)

    def convert_to_tensors(self, dataset):
        """ """
        user = torch.tensor(dataset.user.values).reshape(-1, 1)
        item = torch.tensor(dataset.item.values).reshape(-1, 1)
        label = torch.tensor(dataset.purchase.values).float().reshape(-1, 1)
        return user, item, label

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.label[idx]

    def __len__(self):
        return len(self.user)


if __name__ == "__main__":
    progress_bar = callbacks.RichProgressBar(
        refresh_rate=100,  # number of batches
        leave=True,
        theme=callbacks.progress.rich_progress.RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="#e0bf06",
            progress_bar_pulse="#0606e0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
    )
    dl_param_ = {
        "batch_size": 128,
        "shuffle": False,
        "drop_last": True,
        "num_workers": 6,  # subprocesses to use for data loading
        "pin_memory": False,  # TRUE for GPU
        "prefetch_factor": 100,
    }

    trainer_params = {
        "max_epochs": 5,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "mps",
        "devices": 1,  # number of devices use for training
        "default_root_dir": "./",
        "callbacks": [progress_bar],
    }

    movielens_train_dl = DataLoader(MovieLensDataModule("train"), **dl_param_)
    movielens_val_dl = DataLoader(MovieLensDataModule("val"), **dl_param_)

    reference = MovieLensDataModule("val")
    n_users = reference.n_users
    n_items = reference.n_items

    model_gmf = gmf(
        n_users=n_users,
        n_items=n_items,
        dim=32,
        lr=0.001,
        weight_decay=0,
        gamma=0.1,
        step_size=30,
    )

    trainer_gmf = pl.Trainer(**trainer_params)
    trainer_gmf.fit(
        model=model_gmf,
        train_dataloaders=movielens_train_dl,
        val_dataloaders=movielens_val_dl,
    )

    model_mlp = mlp(
        n_users=n_users,
        n_items=n_items,
        dim_MLP=64,
        dim_GMF=32,
        layers=[32, 32, 32],
        lr=0.001,
        weight_decay=0,
        dropout=False,
        dropout_rate=0.2,
        gamma=0.1,
        step_size=30,
    )

    trainer_mlp = pl.Trainer(**trainer_params)
    trainer_mlp.fit(
        model=model_mlp,
        train_dataloaders=movielens_train_dl,
        val_dataloaders=movielens_val_dl,
    )

    model_neucf = ncf(
        n_users=n_users,
        n_items=n_items,
        layers=[32, 32, 32],
        dim_MLP=64,
        dim_GMF=32,
        pre_trained_GMF=model_gmf,
        pre_trained_MLP=model_mlp,
        lr=0.001,
        weight_decay=0,
        dropout=False,
        dropout_rate=0.2,
        gamma=0.1,
        step_size=30,
    )

    trainer_neumf = pl.Trainer(
        **trainer_params,
    )

    trainer_neumf.fit(
        model=model_neucf,
        train_dataloaders=movielens_train_dl,
        val_dataloaders=movielens_val_dl,
    )
