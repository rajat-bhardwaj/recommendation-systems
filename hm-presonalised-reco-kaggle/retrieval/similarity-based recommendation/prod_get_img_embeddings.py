import sys
import gc

sys.path.append("../")
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import callbacks as callbacks

from Data_pipelines import _read_data_files_helper as helper
from _pytorch_callbacks import MyCallback
from image_classification import ImageDataModule
from image_classification import FeatureExtraction

from torchinfo import summary

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 10)

path_processed_data = Path("")


## load model
class LoadImageClassficationModel(pl.LightningModule):
    def __init__(self, path_saved_model):
        super().__init__()

        self.path_ = path_saved_model

        self.load_pretrained()

    def load_pretrained(self):
        model_ = FeatureExtraction(
            lr=None,
            weight_decay=None,
            gamma=None,
            sc_mode=None,
            sc_patience=None,
            metric=None,
            tune_fc_only=None,
        )
        model_.load_state_dict(
            torch.load(
                self.path_
                / "models_data"
                / "saved_models"
                / "model_image_class_effnetb2.pt"
            )
        )

        fine_tuned_backbone = list(model_.children())[0]
        mlp_top = list(model_.children())[1][:-1]
        self.pretrained_model = nn.Sequential(
            *fine_tuned_backbone, nn.Flatten(), *mlp_top
        )

    def forward(self, x):
        embedding = self.pretrained_model(x)
        return embedding

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image, article_id = batch
        return {article_id: self(image)}


def get_embeddings(preds):
    results = {}

    for key, value in preds.items():
        batch_article_id = key.numpy(force=True)
        batch_embeddings = value.numpy(force=True)

        for i, article_id in tqdm(enumerate(batch_article_id)):
            results.update({str(article_id): batch_embeddings[i]})

    return results


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    gc.collect()

    callbacks_ = MyCallback(None, None, 'min')
    callbacks_ = callbacks_.get_callbacks()[2:]

    img_class_model = LoadImageClassficationModel(path_processed_data.parent)
    for param in img_class_model.parameters():
        param.requires_grad = False

    print(summary(model=img_class_model, input_size=(1, 3, 256, 256)))

    tmdm_tr = ImageDataModule(
        image_path=path_processed_data.parent / "images",
        lookup_path=path_processed_data.parent,
        dataset_type="features",
        frac=0,  # has to be 0 as we will use all available images for feature extraction
    )

    image_dataloader = DataLoader(
        tmdm_tr,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=6,
        prefetch_factor=50
    )

    pl_trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        num_nodes=1,
        strategy="ddp",
        precision="16-mixed",
        enable_model_summary=True,
        enable_progress_bar=True,
        inference_mode=True,
        callbacks=callbacks_,
    )

    predictions = pl_trainer.predict(img_class_model, image_dataloader)

    image_embeddings = {}
    _ = [image_embeddings.update(get_embeddings(batch_)) for batch_ in predictions]

    np.savez(path_processed_data / "prod_img_embeddings", **image_embeddings)
    
    image_embeddings.clear()
    torch.cuda.empty_cache()
