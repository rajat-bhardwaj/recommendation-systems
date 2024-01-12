import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _read_data_files_helper as data_files
import prep_dataset_CF_helper_sequential as cf_helper

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 10)


""" 
## Prepare data for NeuCF
## For each customer, for `leave-one-out evaluation`, 
## exclude items seen in both training and validation sets when sampling negatives.
## STRATEGY ONE: select random n negative examples per customer
"""


def run_neg_strategy_one(path_, path_write, n_neg, strategy, sample=False, recent_tx=True):
    """
    ## setup train and val
    ## train set: 2018 week-38 to 2020 Week-28 (max date: 2020-07-12)
    ## validation set: 2020 Week-29 to 2020 Week-32 (4 weeks of validation data)

    RETURNS
    ---------
    train_: Pandas DataFrame
        Training data
    val_: Pandas DataFrame
        Validation data
    """

    ## setup datasets
    df_transactions = data_files.read_transactions(path_)
    train_df, val_df, _ = data_files.get_model_dfs(
        df_transactions, features=["customer_id", "article_id"]
    )
    train_df["purchase"] = 1
    val_df["purchase"] = 1
    
    ## replacing train_df with training data with recent 10 transactions per customer
    if recent_tx:
        train_df = pd.read_parquet(path_write.parent/'train_tx_last_n_active.parquet')
    
    train_df = train_df.astype("int32")
    val_df = val_df.astype("int32")

    if sample:
        train_df = train_df.sample(frac=0.01)
        val_df = val_df.sample(frac=0.1)

    del df_transactions
    gc.collect()

    if strategy == 2:
        param = {
            "path_features_data": path_write.parent,
            "train_article_tx_features": "train_article_tx_features.parquet",
            "val_article_tx_features": "val_article_tx_features.parquet",
            "quantile": 0.1,
            "n_splits": 10,
        }
        cf_helper.leave_one_out_ds_sequence(
            train_df, val_df, path_write, n_neg, strategy, **param
        )
    else:
        cf_helper.leave_one_out_ds_sequence(
            train_df, val_df, path_write, n_neg, strategy
        )


if __name__ == "__main__":
    
    #set file paths for input and output
    path_model_data = Path("/home/jupyter/reco_project/h-m-data/h-and-m-processed-data/")
    path_base = Path("/home/jupyter/reco_project/h-m-data")

    # set number of negative examples required
    number_negative_eg = 5

    for strategy in [1,2]:
        path_s_neucf = (
            "CF_model_input_SS_neucf_S1" if strategy == 1 else "CF_model_input_SS_neucf_S2"
        )

        run_neg_strategy_one(
            path_base, path_model_data / path_s_neucf, number_negative_eg, strategy
        )
