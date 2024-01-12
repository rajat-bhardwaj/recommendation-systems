import gc
import multiprocessing
import os
import pathlib
from datetime import datetime
from time import time

import numpy as np
import pandas as pd

# PATH_CUSTOMERS = 'customers.csv'
# PATH_TRANSACTIONS = "transactions_train.csv"
# PATH_ARTICLES = "articles.csv"
# PATH_IMAGES = "images"
# PAH_VALN_DATA = "sample_submission.csv"


def customer_lookup(path_base):
    """
    ## one off customer id mapping
    """
    df_customers = pd.read_csv(path_base / "customers.csv")
    lookup_customers = (
        df_customers.customer_id.sort_values()
        .to_frame()
        .reset_index()
        .rename(columns={"index": "_customer_id"})
    )

    lookup_customers.to_csv(path_base / "lookup_table_customers.csv", index=False)


def product_lookup(path_base):
    """
    ## one off article id mapping
    """
    df_product = pd.read_csv(path_base / "articles.csv")
    lookup_product = (
        df_product.article_id.sort_values()
        .to_frame()
        .reset_index()
        .rename(columns={"index": "_article_id"})
    )

    lookup_product.to_csv(path_base / "lookup_table_products.csv", index=False)


def read_customer_file(path_base):
    """
    read csv file from disk, fill missing values
    and setup data types to reduce memory consumption
    """
    df_customers = pd.read_csv(path_base / "customers.csv")
    df_lookup = pd.read_csv(path_base / "lookup_table_customers.csv")

    ## fill missing values
    df_customers["fashion_news_frequency"] = (
        df_customers.fashion_news_frequency.fillna("Unknown")
        .replace(["NONE", "None"], "Unknown")
        .astype("category")
    )

    df_customers["FN"] = df_customers.FN.fillna(0).astype("int8")
    df_customers["Active"] = df_customers.Active.fillna(0).astype("int8")
    df_customers["club_member_status"] = df_customers.club_member_status.fillna(
        "UNKNOWN"
    ).astype("category")

    ## median age
    median_age = df_customers.age.median()
    df_customers["age"] = df_customers.age.fillna(median_age).astype("float32")

    ## reduce size and set data types
    df_customers = (
        df_customers.merge(df_lookup, on="customer_id")
        .drop(columns=["customer_id"])
        .rename(columns={"_customer_id": "customer_id"})
        .astype({"customer_id": "int64"})
    )

    df_customers["postal_code"] = df_customers.postal_code.apply(
        lambda x: int(x[-16:], base=16)
    )

    return df_customers


def read_articles(path_base):
    """
    Reads csv file from disk
    reduce df size by manipulating the data types
    """
    columns_categorical = [
        "prod_name",
        "product_type_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "index_code",
    ]
    datatypes = {column: "category" for column in columns_categorical}

    df_articles = pd.read_csv(path_base / "articles.csv", dtype=datatypes)
    df_lookup = pd.read_csv(path_base / "lookup_table_products.csv")

    df_articles = (
        df_articles.merge(df_lookup, on="article_id")
        .drop(columns=["article_id"])
        .rename(columns={"_article_id": "article_id"})
        .astype({"article_id": "int64"})
    )

    return df_articles


def read_transactions(path_base):
    """
    Reads csv file from disk
    reduce df size by manipulating the data types
    """
    df_transactions = pd.read_csv(
        path_base / "transactions_train.csv",
        dtype={"sales_channel_id": "category", "price": "float32"},
    )

    df_lookup_cust = pd.read_csv(path_base / "lookup_table_customers.csv")
    df_lookup_prod = pd.read_csv(path_base / "lookup_table_products.csv")

    df_transactions = (
        df_transactions.merge(df_lookup_cust, on="customer_id")
        .drop(columns=["customer_id"])
        .rename(columns={"_customer_id": "customer_id"})
        .astype({"customer_id": "int64"})
    )

    df_transactions = (
        df_transactions.merge(df_lookup_prod, on="article_id")
        .drop(columns=["article_id"])
        .rename(columns={"_article_id": "article_id"})
        .astype({"article_id": "int64"})
    )

    df_transactions["t_dat"] = pd.to_datetime(df_transactions.t_dat)

    # Introduce year partition and season
    df_transactions["tx_month"] = df_transactions.t_dat.dt.month.astype(np.int8)
    df_dates = df_transactions.t_dat.dt.isocalendar().drop(columns=["day"])
    df_transactions = (
        pd.concat([df_transactions, df_dates], axis=1)
        .rename(columns={"year": "tx_year", "week": "tx_week"})
        .astype({"tx_year": "category", "tx_week": np.int8})
    )

    del df_dates

    return df_transactions


def get_model_dfs(df_transactions, features=["customer_id", "article_id", "t_dat"]):
    """ """

    ## fix all the dates as defined earlier for development and test set
    min_date_ = df_transactions.t_dat.min()
    max_date_train_df = df_transactions[
        (df_transactions.tx_year == 2020) & (df_transactions.tx_week == 28)
    ].t_dat.max()
    max_date_val_df = df_transactions[
        (df_transactions.tx_year == 2020) & (df_transactions.tx_week == 32)
    ].t_dat.max()
    max_date_test_df = df_transactions[
        (df_transactions.tx_year == 2020) & (df_transactions.tx_week == 39)
    ].t_dat.max()

    # fiilter datasets on date
    train_df = (
        df_transactions[df_transactions.t_dat.between(min_date_, max_date_train_df)]
        .filter(features)
        .drop_duplicates()
    )

    val_df = (
        df_transactions[
            df_transactions.t_dat.between(
                (max_date_train_df + pd.DateOffset(days=1)), max_date_val_df
            )
        ]
        .filter(features)
        .drop_duplicates()
    )

    test_df = (
        df_transactions[
            df_transactions.t_dat.between(
                (max_date_val_df + pd.DateOffset(days=1)), max_date_test_df
            )
        ]
        .filter(features)
        .drop_duplicates()
    )

    return train_df, val_df, test_df

def findAllFiles(data_dir, plot=True):
    """
    Args:
    data_dir: filepath / poxis

    returns:
    pandas dataframe
    """
    image_count = list(data_dir.glob("*/*.*"))
    image_count_df = pd.DataFrame(image_count, columns=["filepath"])
    image_count_df["class_name"] = image_count_df.filepath.apply(
        lambda x: x.parts[-2])
    image_count_df["article_id"] = image_count_df.filepath.apply(
        lambda x: x.parts[-1].split(".")[0])
    image_count_df["file_extension"] = image_count_df.filepath.apply(
        lambda x: x.parts[-1].split(".")[-1])
    image_count_df = image_count_df[image_count_df.file_extension.isin(
        ["jpeg", "jpg", "png"])]
    image_count_group = (image_count_df.groupby(
        ["class_name"]).filepath.count().to_frame())
    
    if plot:
        print(image_count_group.plot.bar(figsize=(20, 6)))

    return image_count_df
