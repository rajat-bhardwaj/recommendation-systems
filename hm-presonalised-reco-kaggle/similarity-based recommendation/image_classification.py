"""
Fine tune a pre-trained image model such that the image embeddings are 
more representative of the garments. This is achieved by running a classfication task
to predct the `garment_group_name`.

"""
import os
import sys
import gc
import warnings

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
sys.path.append("../")
warnings.filterwarnings("ignore")

from time import time
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl
from lightning.pytorch import callbacks as callbacks
from lightning.pytorch.loggers import TensorBoardLogger

from torchvision.transforms import v2
from torchvision.io import read_image
import torchvision.models as models

from torchmetrics.classification import MulticlassAccuracy

from Data_pipelines import _read_data_files_helper as helper
from _pytorch_callbacks import MyCallback
from _pytorch_callbacks import get_logger

IMAGE_SIZE = 256


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self, image_path, lookup_path, dataset_type="train", frac=0.7, sample=False
    ):
        super().__init__()

        self.image_path = image_path
        self.lookup_path = lookup_path
        self.dataset_type = dataset_type
        self._image_size = (IMAGE_SIZE, IMAGE_SIZE)
        self.frac = frac
        self.sample = sample

        self.prepare_pandas_df()
        self.create_train_trnsfrms()
        self.create_val_transform()

    def prepare_pandas_df(self):
        pd_image_df = helper.findAllFiles(self.image_path, plot=False).astype(
            {"article_id": "int32"}
        )
        df_lookup = pd.read_csv(self.lookup_path / "lookup_table_products.csv").astype(
            "int32"
        )

        image_df = (
            pd_image_df.merge(df_lookup, on="article_id")
            .drop(columns=["article_id"])
            .rename(columns={"_article_id": "article_id"})
        )
        assert image_df.shape[0] == pd_image_df.shape[0]

        prod_meta = (
            helper.read_articles(self.lookup_path)
            .filter(["article_id", "garment_group_name"])
            .astype({"article_id": "int32", "garment_group_name": "str"})
        )

        prod_meta["labels"] = prod_meta.garment_group_name.apply(
            lambda x: str(x).strip().lower()
        )
        prod_meta["labels"] = prod_meta.labels.replace("\\W+", "", regex=True)
        prod_meta["labels"] = prod_meta.garment_group_name.factorize()[0]

        image_df = image_df.merge(prod_meta, on="article_id").filter(
            ["article_id", "filepath", "labels"]
        )

        image_df["filepath"] = image_df.filepath.apply(lambda x: str(x))

        if self.sample:
            image_df = image_df.sample(frac=0.01, random_state=42, axis=0)

        if self.frac == 0:
            self.train = image_df
        else:
            self.train = image_df.sample(frac=self.frac, random_state=42, axis=0)
            self.val = image_df.drop(index=self.train.index)

    def __len__(self):
        if self.dataset_type in ["train", "features"]:
            len__ = len(self.train)
        elif self.dataset_type == "val":
            len__ = len(self.val)
        return len__

    def create_train_trnsfrms(self):
        """ """
        self.list_transfrms = [
            v2.RandomCrop(
                size=self._image_size, pad_if_needed=True, padding_mode="reflect"
            ),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=(-120, 120)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.RandomInvert(p=0.3),
            v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.RandomPosterize(bits=2),
            v2.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.1, hue=0.5),
        ]

        self.tr_trnsfrm_apply = v2.Compose(self.list_transfrms)
        self.tr_transfrm_rand = v2.RandomApply(self.list_transfrms)
        self.tr_transfrm_auto = v2.AutoAugment()

    def create_val_transform(
        self,
    ):
        """ """
        self.val_transfrm = v2.Compose(
            [
                v2.Resize(size=self._image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def run_train_trnsfrm(self, image):
        image = v2.Resize(size=self._image_size, antialias=True)(image)
        image = self.tr_transfrm_auto(image)  # auto or random
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            image
        )

        return image

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            image = read_image(self.train.iloc[idx]["filepath"])
            image = self.run_train_trnsfrm(image)
            label = self.train.iloc[idx]["labels"]
        elif self.dataset_type == "val":
            image = read_image(self.val.iloc[idx]["filepath"])
            image = self.val_transfrm(image)
            label = self.val.iloc[idx]["labels"]
        elif (self.dataset_type == "features") & (self.frac == 0):
            image = read_image(self.train.iloc[idx]["filepath"])
            image = self.val_transfrm(image)
            article_ids = self.train.iloc[idx]["article_id"]
            return image, article_ids

        return image, label


class FeatureExtraction(pl.LightningModule):
    def __init__(
        self,
        lr,
        weight_decay,
        gamma,
        sc_mode,
        sc_patience,
        metric,
        tune_fc_only=True,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.sc_mode = sc_mode
        self.sc_patience = sc_patience
        self.metric = metric

        num_target_classes = 21

        self.example_input_array = torch.zeros(
            32, 3, IMAGE_SIZE, IMAGE_SIZE
        )  # image shape

        model = models.efficientnet_b2(weights="IMAGENET1K_V1")
        layers = list(model.children())[:-1]
        num_filters = model.classifier[1].in_features  # 1408
        self.feature_extractor = nn.Sequential(*layers)

        # option to only tune the fully-connected layers
        if tune_fc_only:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:  # set last block for fine tuning
            for name, param in self.feature_extractor.named_parameters():
                if int(name.split(".")[1]) < 7:
                    param.requires_grad = False

        self.top_layer = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.BatchNorm1d(
                num_filters // 2,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),  # Block 1
            nn.Linear(num_filters // 2, num_filters // 4),
            nn.ReLU(),
            nn.BatchNorm1d(
                num_filters // 4,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),  # Block 2
            nn.Linear(num_filters // 4, num_target_classes),  # Block 3
        )

        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.loss_fn = nn.CrossEntropyLoss()
        # create accuracy metric
        self.acc = MulticlassAccuracy(num_classes=num_target_classes, average="macro")

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        representations = self.top_layer(representations)
        return representations

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
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    gc.collect()

    path_processed_data = Path("")

    path_model = path_processed_data / "models_data"

    config = {
        "lr": 0.001,
        "weight_decay": 0,
        "gamma": 0.1,
        "max_epochs": 10,
        "batch_size": 32,
        "tune_fc_only": False,
        "frac": 0.7,
        "sample": False,  # only return 1% data (images) split into train and val
        "metric_es": "val_acc",
        "metric_sc": "train_loss_epoch",
        "mode_es": "max",
        "mode_sc": "min",
        "patience_es": 5,  # callbacks; early stopping
        "patience_sc": 2,  # scheduler ReduceLROnPlateau patience
        "load_saved": True,
        "save_model": True,
        "run_as_test": False,  # Load all data and limit number of train and val batched
    }

    callbacks_ = MyCallback(
        config["metric_es"], config["patience_es"], config["mode_es"]
    )
    callbacks_ = callbacks_.get_callbacks()

    logger = TensorBoardLogger(
        save_dir=path_model,
        log_graph=True,
        name="lightning_logs",
        version="img_eff_netb2_{}".format(str(int(time()))),
    )

    trainer_params = {
        "max_epochs": config["max_epochs"],
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "cuda",
        "devices": 1,
        "num_nodes": 1,
        "strategy": "ddp",
        "precision": "16-mixed",
        "logger": logger,
    }

    dl_param_ = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": True,
        "num_workers": 5,
        "pin_memory": True,
        "pin_memory_device": "cuda",
        "prefetch_factor": 500,
    }

    tmdm_tr = ImageDataModule(
        path_processed_data.parent / "images",
        path_processed_data.parent,
        "train",
        config["frac"],
        config["sample"],
    )

    if config["run_as_test"]:
        test_config = {
            # "log_every_n_steps": 10,
            "limit_train_batches": 500,
            "limit_val_batches": 50,
            # "profiler": "pytorch",
        }
        trainer_params.update(test_config)

    tr_image_dl = DataLoader(tmdm_tr, **dl_param_)

    tmdm_val = ImageDataModule(
        path_processed_data.parent / "images",
        path_processed_data.parent,
        "val",
        config["frac"],
        config["sample"],
    )

    dl_param_.update({"shuffle": False})
    val_image_dl = DataLoader(tmdm_val, **dl_param_)

    model_ = FeatureExtraction(
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        gamma=config["gamma"],
        sc_mode=config["mode_sc"],
        sc_patience=config["patience_sc"],
        metric=config["metric_sc"],
        tune_fc_only=config["tune_fc_only"],
    )

    if config["load_saved"]:
        model_.load_state_dict(
            torch.load(path_model / "saved_models" / "model_image_class_effnetb2.pt")
        )

    trainer_gmf = pl.Trainer(callbacks=callbacks_, **trainer_params)
    trainer_gmf.fit(
        model=model_, train_dataloaders=tr_image_dl, val_dataloaders=val_image_dl
    )

    if config["save_model"]:
        torch.save(
            model_.state_dict(),
            path_model / "saved_models" / "model_image_class_effnetb2.pt",
        )

    torch.cuda.empty_cache()
