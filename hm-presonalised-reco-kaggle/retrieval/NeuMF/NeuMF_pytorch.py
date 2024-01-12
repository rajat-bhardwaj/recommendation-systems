import warnings
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import callbacks as callbacks
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def return_mlp(layers, dim_MLP, dropout_rate):
    # As both user and item embeddings are concatenate
    layers[0] = 2 * dim_MLP

    # setup MLP layers
    mlp_layers = nn.ModuleList()
    for i, (in_feat, out_feat) in enumerate(zip(layers[:-1], layers[1:])):
        mlp_layers.append(nn.Linear(in_feat, out_feat))
        if i in [0, 1, 2]:
            mlp_layers.append(
                nn.BatchNorm1d(out_feat, affine=False, track_running_stats=False)
            )
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_rate))

    return mlp_layers


class DataModuleClass(pl.LightningDataModule):
    def __init__(
        self,
        path_write,
        strategy,
        sample,
        sample_frac_tr,
        subsample_neg,
        _num_neg,
        dataloader_params,
    ):
        """
        PARAMETERS
        ----------
        path_write: Posix
            Path to processed data (from data pipeline)
        strategy: int
            Negative sampling strategy. Available values; 1,2,3
        sample: bool
            Generate a small sample of data
        sample_frac_tr: float
            fraction of data to be used for sampling.
            Validation set will contain all customer data from training
        ds_type: String
            Type of dataset. "train" and "val"
        subsample_neg: bool
            Subsample the negative
        _num_ne: int
            Number of negatives for each customer
        """
        super().__init__()

        self.path_write = path_write
        self.strategy = strategy
        self.sample = sample
        self.sample_frac_tr = sample_frac_tr
        # subsample_neg; change to True or use param to reduce
        # negative example per customer at run time
        self.subsample_neg = subsample_neg
        self._num_neg = _num_neg
        self.dataloader_params = dataloader_params

        self.folder_s1 = "CF_model_input_neucf_S1"
        self.folder_s2 = "CF_model_input_neucf_S2"
        self.folder_s3 = "CF_model_input_neucf_S3"

    def subsample_negatives(self, dataset_):
        """ """
        positives = dataset_[dataset_.purchase == 1]
        negatives = dataset_[dataset_.purchase == 0]

        groups_ = negatives.groupby(["customer_id"])

        df_list = []
        for grp in tqdm(groups_):
            df_list.append(
                [
                    grp[0],
                    grp[1].article_id.sample(
                        n=self._num_neg, replace=self.sample, random_state=42
                    ),
                ]
            )

        test_ = pd.DataFrame(df_list, columns=["customer_id", "article_id"])
        df_list.clear()
        test_ = test_.explode(["article_id"])
        test_["purchase"] = 0

        if not self.sample:
            # taking a sample from validation set may cause
            # some customers to only have either only positive example
            # or negative examples
            assert int(test_.shape[0] / self._num_neg) == dataset_.customer_id.nunique()

        df_final = pd.concat([positives, test_]).astype("int32")

        return df_final

    def custom_ds_prep(self, foldername):
        """
        ## setup train and validation dataset for Strategy One and Two

        PARAMETERS
        ----------
        foldername: String
            Folder name of train and val datasets

        RETURNS
        ---------
        train_: Pandas DataFrame
            Training data
        val_: Pandas DataFrame
            Validation data
        """

        train_ = pd.read_parquet(
            self.path_write
            / foldername
            / "train_ds_sample_stg_{}.parquet".format(self.strategy)
        ).astype("int32")
        val_ = pd.read_parquet(
            self.path_write
            / foldername
            / "val_ds_sample_stg_{}.parquet".format(self.strategy)
        ).astype("int32")

        return train_, val_

    def convert_to_tensors(self, dataset):
        """ """
        user = torch.tensor(dataset.customer_id.values).reshape(-1, 1)
        item = torch.tensor(dataset.article_id.values).reshape(-1, 1)
        label = torch.tensor(dataset.purchase.values).float().reshape(-1, 1)

        return user, item, label

    def prepare_data(self):
        """
        Prepare training and validation dataset based on the negative sampling strategy
        """
        if self.strategy == 1:
            self.train_, self.val_ = self.custom_ds_prep(self.folder_s1)

        elif self.strategy == 2:
            self.train_, self.val_ = self.custom_ds_prep(self.folder_s2)

        elif self.strategy == 3:
            self.train_, self.val_ = self.custom_ds_prep(self.folder_s3)

        if self.sample:
            customers = self.train_.customer_id.drop_duplicates().sample(
                frac=self.sample_frac_tr, replace=False, random_state=42
            )

            self.train_ = self.train_[self.train_.customer_id.isin(customers)]
            # self.val_ = self.val_[self.val_.customer_id.isin(customers)]

        # remove products from validation that are not in train dataset
        prod = self.train_.article_id.unique()
        self.val_ = self.val_[self.val_.article_id.isin(prod)]

        if self.subsample_neg:
            self.train_ = self.subsample_negatives(self.train_)
            self.val_ = self.subsample_negatives(self.val_)

    def setup(self, stage):
        if stage == "fit":
            user, item, label = self.convert_to_tensors(self.train_)
            self.train_data = TensorDataset(user, item, label)

            user_, item_, label_ = self.convert_to_tensors(self.val_)
            self.val_data = TensorDataset(user_, item_, label_)

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.val_data, **self.dataloader_params)


class GeneralisedMatrixFactorisation(pl.LightningModule):
    def __init__(
        self, n_users, n_items, dim, lr=1e-3, weight_decay=0, gamma=0.1, step_size=3
    ):
        """
        Generalised Matrix Factorisation

        PARAMETERS
        ----------
        n_users: int
            Number of users in the dataset
        n_items: int
            Number of items
        dim: int
            Dimension of embeddings
        lr: float
            Learning rate
        weight_decay: int
            Regularisation; weight_decay > 0 introduces L2 Penalty
        gamma: float
            Exponential decay rate
        """
        super().__init__()

        # by default weights = True
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=dim, max_norm=1
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=dim, max_norm=1
        )

        torch.nn.init.uniform_(self.user_embedding.weight)
        torch.nn.init.uniform_(self.item_embedding.weight)

        # Generalised using linear layer than just dot product
        self.output = nn.Linear(in_features=dim, out_features=1)

        # Binary Cross Entropy between the target and the input probabilities:
        self.loss_ = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.step_size = step_size

    def forward(self, user_, item_):
        user_embedding = self.user_embedding(user_)
        item_embedding = self.item_embedding(item_)

        # element-wise product of vectors
        mul = torch.mul(user_embedding, item_embedding)
        mul = self.output(mul)

        return mul

    def configure_optimizers(self):
        # weight_decay > 0 introduces L2 Penalty
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def _step(self, batch):
        user, item, label = batch
        pred_label = self(user, item)
        # print("---- ", pred_label.shape, label.shape)
        loss = self.loss_(pred_label, label.unsqueeze(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # perform logging
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user, item = batch
        return self(user, item)


class MultiLayerPerceptron(pl.LightningModule):
    """ """

    def __init__(
        self,
        n_users,
        n_items,
        dim_MLP,
        layers,
        lr=1e-5,
        weight_decay=0,
        dropout=False,
        dropout_rate=0.2,
        batchnorm=True,
        gamma=0.1,
        step_size=3,
    ):
        """
        MultiLayer Perceptron

        PARAMETERS
        ----------
        n_users: int
            Number of users in the dataset
        n_items: int
            Number of items
        dim_MLP: int
            Dimension of MLP embeddings
        layers: list
            list of units (int) per layer
        lr: float
            Learning rate
        weight_decay: int
            Regularisation; weight_decay > 0 introduces L2 Penalty
        dropout: bool
            to include dropout or not
        dropout_rate: float
            dropout_rate
        batchnorm: bool
            whether to include batch normalisation or not
        gamma: float
            LR step rate
        step_size: int
            step size for LR step scheduler
        """

        ### dim and layers[0] should be of same size
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=dim_MLP, max_norm=1
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=dim_MLP, max_norm=1
        )

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # MLP parameters
        self.layers = layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm

        # As both user and item embeddings are concatenate
        self.layers[0] = 2 * dim_MLP
        self.mlp_layers = return_mlp(self.layers, dim_MLP, self.dropout_rate)
        self.mlp_layers.append(nn.Linear(self.layers[-1], 1))

        # # setup MLP layers
        # # append last layer with one output
        # # the last layer should match embedding length
        # # of GMF and NETWORK output of 2nd last layer.

        ## setup loss function and optimiser
        self.loss_ = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.step_size = step_size

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def forward(self, user_, item_):
        user_embedding = self.user_embedding(user_)
        item_embedding = self.item_embedding(item_)

        # print(" **** --- **** ", user_embedding.shape, item_embedding.shape)
        emb_concat = torch.concatenate([user_embedding, item_embedding], dim=-1)
        for index, layer_ in enumerate(self.mlp_layers):
            emb_concat = layer_(emb_concat)

        return emb_concat

    def _step(self, batch, prefix):
        user, item, label = batch
        pred_label = self(user, item)
        # print("---- ", pred_label.shape, label.shape)
        loss = self.loss_(pred_label, label.unsqueeze(-1))
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True if prefix == "train" else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True if prefix == "val" else False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user, item = batch
        return self(user, item)


class NeuMatrixFactorisation(pl.LightningModule):
    """
    Neural Matrix Factorisation combining
    Generalised Matrix Factorisation and Multilayer perceptron
    """

    def __init__(
        self,
        n_users,
        n_items,
        layers,
        dim_GMF,
        dim_MLP,
        pre_train,
        pre_trained_GMF,
        pre_trained_MLP,
        lr=1e-3,
        weight_decay=0,
        dropout=False,
        dropout_rate=0.2,
        batchnorm=True,
        gamma=0.1,
        step_size=3,
    ):
        """
        PARAMETERS
        ----------
        n_users: int
            Number of users in the dataset
        n_items: int
            Number of items
        layers: list
            list of units (int) per layer
        dim_GMF: int
            Dimension of GMF embeddings
        dim_MLP: int
            Dimension of MLP embeddings
        pre_trained_GMF: lightning model
            pre trained GMF model component
        pre_trained_MLP: lightning model
            pre trained MLP model component
        lr: float
            Learning rate
        weight_decay: int
            Regularisation; weight_decay > 0 introduces L2 Penalty
        dropout: bool
            to include dropout or not
        dropout_rate: float
            dropout_rate
        gamma: float
            Exponential decay rate
        """
        super().__init__()

        # self.pre_trained_GMF = pre_trained_GMF
        # self.pre_trained_MLP = pre_trained_MLP

        # initialise embeddings
        self.user_embedding_GMF = nn.Embedding(
            num_embeddings=n_users, embedding_dim=dim_GMF
        )
        self.item_embedding_GMF = nn.Embedding(
            num_embeddings=n_items, embedding_dim=dim_GMF
        )
        self.user_embedding_MLP = nn.Embedding(
            num_embeddings=n_users, embedding_dim=dim_MLP
        )
        self.item_embedding_MLP = nn.Embedding(
            num_embeddings=n_items, embedding_dim=dim_MLP
        )

        nn.init.uniform_(self.user_embedding_GMF.weight)
        nn.init.uniform_(self.item_embedding_GMF.weight)
        nn.init.xavier_normal_(self.user_embedding_MLP.weight)
        nn.init.xavier_normal_(self.item_embedding_MLP.weight)

        # create MLP model
        # MLP parameters
        self.layers = layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm

        # As both user and item embeddings are concatenate
        self.layers[0] = 2 * dim_MLP
        self.mlp_layers = return_mlp(self.layers, dim_MLP, self.dropout_rate)

        ## Keeping this dimension same here and in MLP as
        ## we are using pre-trained weights for warm-start
        self.last_layer_size = dim_GMF + self.layers[-1]
        self.mlp_layers.append(nn.Linear(self.last_layer_size, 1))

        ## setup loss function and optimiser
        self.loss_ = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.step_size = step_size

        if pre_train:
            ## initialise weights for fine tuning
            self.initialise_weights(pre_trained_GMF, pre_trained_MLP)

    def initialise_weights(self, pre_trained_GMF, pre_trained_MLP):
        ## initialise weights from pre trained model
        self.user_embedding_GMF.weight.data.copy_(pre_trained_GMF.user_embedding.weight)
        self.item_embedding_GMF.weight.data.copy_(pre_trained_GMF.item_embedding.weight)
        self.user_embedding_MLP.weight.data.copy_(pre_trained_MLP.user_embedding.weight)
        self.item_embedding_MLP.weight.data.copy_(pre_trained_MLP.item_embedding.weight)

        for index, layer in enumerate(pre_trained_MLP.mlp_layers[:-1]):
            if layer == nn.Linear:
                self.mlp_layers[index].weight.data.copy_(layer.weight)
                self.mlp_layers[index].bias.data.copy_(layer.bias)

        # the last layer will not get the weight and bias of MLP last layer
        # since the MLP layer dimension is smaller than NeuMF last layers
        # due to contactination of GML AND MLP enbeddings.
        # another approach would be to use average or weighted
        # weight and bias from both the last layers of MLP and GMF. TBD

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]
        # return optimizer

    def forward(self, user_, item_):
        # MF
        user_embedding_GMF = self.user_embedding_GMF(user_)
        item_embedding_GMF = self.item_embedding_GMF(item_)
        # results in 2*dim
        matrix_factorisation = torch.mul(user_embedding_GMF, item_embedding_GMF)

        # MLP
        user_embedding_MLP = self.user_embedding_MLP(user_)
        item_embedding_MLP = self.item_embedding_MLP(item_)

        emb_concat = torch.cat([user_embedding_MLP, item_embedding_MLP], dim=-1)
        for index, layer_ in enumerate(self.mlp_layers[:-1]):
            emb_concat = layer_(emb_concat)

        ## add output from both models (GMF and MLP)
        ## emb_concat dim is same as 2nd last layer
        concat_outputs = torch.cat([matrix_factorisation, emb_concat], dim=-1)

        # print(" --- forward test", matrix_factorisation.shape, emb_concat.shape, concat_outputs.shape)
        concat_outputs = self.mlp_layers[-1](concat_outputs)

        return concat_outputs

    def _step(self, batch, prefix):
        user, item, label = batch
        pred_label = self(user, item)
        # print("---- ", pred_label.shape, label.shape)
        loss = self.loss_(pred_label, label.unsqueeze(-1))
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True if prefix == "train" else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True if prefix == "val" else False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, "test")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user, item = batch
        pred = self(user, item)
        return pred
