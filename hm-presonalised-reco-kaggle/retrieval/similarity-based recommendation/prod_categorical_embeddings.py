""" 
Train a pytorch model to generate embedidngs from product categorical data.
A simple approach is to use a Sentence transformer with embedding model.

Here I have tuned the embeddings towards predicting `garment_group_name`
therfore the embedding vector is representative of the garment group.


"""
import sys

sys.path.append("../")

import gc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from time import time
import pandas as pd
import numpy as np
import string

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl
from lightning.pytorch import callbacks as callbacks
from lightning.pytorch.loggers import TensorBoardLogger

from Data_pipelines import _read_data_files_helper as helper

from _pytorch_callbacks import MyCallback

from torchmetrics.classification import MulticlassAccuracy
from torchinfo import summary


class StaticFeaturesDataModule(pl.LightningDataModule):
    def __init__(self, path_, dataloader_params, frac=0.7, sample=True):
        super().__init__()

        self.path_ = path_
        self.s_punct = set(string.punctuation)
        self.rnd = 42
        self.sample = sample
        self.frac = frac
        self.dataloader_params = dataloader_params
        self.target_column = "garment_group_name"

        self.val_dl_config = self.dataloader_params.copy()
        self.val_dl_config.update({"shuffle": False})

        self.prepare_dataset()

    def clean(self, x):
        txt = str(x).lower().strip().replace("/", " ")
        txt = "".join([word for word in txt if word not in self.s_punct])
        return txt

    def prepare_dataset(self):
        prod_meta = helper.read_articles(self.path_)
        prod_meta = prod_meta.drop(columns=["detail_desc"])
        cl_prod_meta = prod_meta.set_index(["article_id"])
        cl_prod_meta = cl_prod_meta.applymap(lambda x: self.clean(x))

        ## create index for each categorical value
        self.product_features = cl_prod_meta.apply(lambda x: x.factorize()[0]).astype(
            "int32"
        )

        if self.sample:
            self.product_features = self.product_features.sample(
                frac=0.01, random_state=self.rnd, axis=0
            )

        _columns = self.product_features.columns
        self.input_cols = _columns[
            ~_columns.isin(["garment_group_no", self.target_column])
        ]

        # split train and val
        self.train = self.product_features.sample(
            frac=self.frac, random_state=self.rnd, axis=0
        )
        self.val = self.product_features.drop(index=self.train.index)

    def get_embedding_shape(self, dim):
        embedding_shape = []
        for col in self.input_cols:
            embedding_shape.append((self.product_features[col].nunique(), dim))

        nclasses = self.product_features[self.target_column].nunique()

        return embedding_shape, nclasses

    def setup(self, stage=None):
        if stage == "fit":
            tr_features = torch.as_tensor(self.train.filter(self.input_cols).values)
            tr_label = torch.as_tensor(self.train.filter([self.target_column]).values.flatten()).type(torch.LongTensor)
            self.train_data = TensorDataset(tr_features, tr_label)

            val_features = torch.as_tensor(self.val.filter(self.input_cols).values)
            val_label = torch.as_tensor(self.val.filter([self.target_column]).values.flatten()).type(torch.LongTensor)
            self.val_data = TensorDataset(val_features, val_label)

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.val_data, **self.val_dl_config)


class TextEmbeddings(pl.LightningModule):
    def __init__(
        self,
        embedding_shape,
        nclasses,
        lr,
        weight_decay,
        gamma,
        metric,
        sc_patience,
        sc_mode,
    ):
        super().__init__()
        
        embedding_list = []
        for num, dim in embedding_shape:
            emb = nn.Embedding(num, dim)
            nn.init.xavier_normal_(emb.weight)
            embedding_list.append(emb)
        
        self.all_embeddings = nn.ModuleList(embedding_list)
        
        embedding_list.clear()

        initial_dim = sum([dim for _, dim in embedding_shape])
        all_layers = [
            nn.Linear(in_features=initial_dim, out_features=initial_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(
                num_features=initial_dim // 2,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True),  ## block 1
            nn.Linear(in_features=initial_dim // 2, out_features=nclasses),
        ]

        self.topmlp = nn.Sequential(*all_layers)
        
        all_layers.clear()

        self.metric = metric
        self.sc_patience = sc_patience
        self.sc_mode = sc_mode
        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.loss_fn = nn.CrossEntropyLoss()
        # create accuracy metric
        self.acc = MulticlassAccuracy(num_classes=nclasses, average="macro")

    def forward(self, x):
        ## embedding_shape and model input should have the same feature/column order
        ## extract the vector at the feature index ## STRING LOOKUP
        embeddings = []
        for index, feat_embedding in enumerate(self.all_embeddings):
            embeddings.append(feat_embedding(x[:,index]))
            
        output = torch.cat(embeddings, 1)
        output = self.topmlp(output)

        return output

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, factor=self.gamma, patience=self.sc_patience, mode=self.sc_mode
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.metric,
        }

    def _step(self, batch, prefix):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True if prefix == "train" else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,  # single GPU
        )
        self.log(
            f"{prefix}_acc",
            acc,
            on_step=True if prefix == "train" else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,  # single GPU
        )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch, "val")
        return loss, acc


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    path_processed_data = Path("")
    path_model = path_processed_data / "models_data"

    config = {
        "dim": 8,
        "lr": 0.0001,
        "weight_decay": 0,
        "gamma": 0.1,
        "max_epochs": 3,
        "batch_size": 32,
        "frac": 0.7,
        "sample": False,  # only return 1% data (images) split into train and val
        "metric_es": "val_acc",
        "metric_sc": "train_loss_epoch",
        "mode_es": "max",
        "mode_sc": "min",
        "patience_es": 5,  # callbacks; early stopping
        "patience_sc": 2,  # scheduler ReduceLROnPlateau patience
    }

    callbacks_ = MyCallback(
        config["metric_es"], config["patience_es"], config["mode_es"]
    )
    callbacks_ = callbacks_.get_callbacks()

    logger = TensorBoardLogger(
        save_dir=path_model,
        log_graph=True,
        name="lightning_logs",
        version="txt_embedding_{}".format(str(int(time()))),
    )

    trainer_params = {
        "max_epochs": config["max_epochs"],
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "accelerator": "cuda",
        "strategy": "ddp",
        "devices": 1,
        "num_nodes": 1,
        "precision": "16-mixed",
        "logger": logger,
        # "limit_train_batches": 100,
        # "limit_val_batches": 100,
    }

    dl_param_ = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": True,
        "num_workers": 5,
        "pin_memory": True,
        "pin_memory_device": "cuda",
        "prefetch_factor": 100,
    }

    dm = StaticFeaturesDataModule(
        path_processed_data.parent, dl_param_, config["frac"], config["sample"]
    )
    
    embedding_shape, nclasses = dm.get_embedding_shape(config["dim"])

    model_ = TextEmbeddings(
        embedding_shape,
        nclasses,
        config["lr"],
        config["weight_decay"],
        config["gamma"],
        config["metric_sc"],
        config["patience_sc"],
        config["mode_sc"],
    )

    trainer_ = pl.Trainer(callbacks=callbacks_, **trainer_params)
    trainer_.fit(model=model_, datamodule=dm)
    
    torch.save(
            model_.state_dict(),
            path_model / "saved_models" / "model_cat_feat_embeddings_class.pt",
        )
