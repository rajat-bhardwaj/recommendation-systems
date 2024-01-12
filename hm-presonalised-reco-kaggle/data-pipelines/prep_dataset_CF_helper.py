import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
pandarallel.initialize(nb_workers=7, progress_bar=True, verbose=2, use_memory_fs=False)

warnings.filterwarnings("ignore")
tqdm.pandas(desc="Negative samples!")


def _get_random_(x, n_neg, replace=False):
    return np.random.choice(x, n_neg, replace=replace)


def gen_negative_examples(new_df, n_neg, test=True):
    """ """

    def _foo(feature_name):
        if test:
            dataframe_ = new_df.filter(["customer_id", feature_name])
        else:
            dataframe_ = new_df

        dataframe_["article_id"] = dataframe_[feature_name].parallel_apply(
            lambda x: _get_random_(x, n_neg)
        )
        dataframe_ = dataframe_.drop(columns=[feature_name])
        dataframe_ = dataframe_.explode("article_id")
        dataframe_["purchase"] = 0

        return dataframe_

    if test:
        val_df = _foo("avail_products_val")
        train_df = _foo("avail_products_train")
        return train_df, val_df
    else:
        return _foo("article_id_tr")


def get_prod_candidates(x, prod_train, prod_val) -> list:
    """
    Runs union of interacted items for customers both in training and validation set.
    Computes available products for further sampling.

    Parameters
    ----------
    x: Pandas DataFrame row
    prod_train : numpy array
        Products available in training dataset
    prod_val: numpy array
        Products available in validation dataset

     Returns
    -------
    list
        Returns a list of numpy array for both training and validation dataset
    """
    union_ = np.union1d(x.article_id_tr, x.article_id_val)

    mask_train = np.isin(prod_train, union_, assume_unique=True)
    avail_products_train = prod_train[~mask_train]

    mask_val = np.isin(prod_val, union_, assume_unique=True)
    avail_products_val = prod_val[~mask_val]

    del (union_, mask_train, mask_val)

    return [avail_products_train, avail_products_val]


def compute_engagement(input_df, quantile) -> np.array:
    """
    Generates an engageent score to identify the lowest performing product
    to sample the negative examples.

    Parameters
    ----------
    input_df: Pandas DataFrame
        Input datafarme with article features
    quantile: float
        Quantile fraction

    Returns
    -------
    Numpy array of products
    """

    ## interaction
    ## engagement = (unique customers count * total purchases / article availability)
    ## overall average customer in dataset
    ## recent customer engagement 3 months

    df_engagement = input_df.filter(
        [
            "article_id",
            "u_customers_lt",
            "t_purchases_lt",
            "article_availability_lt",
            "u_customers_3m",
            "t_purchases_3m",
            "article_availability_3m",
        ]
    )

    # if article_availability_3m is 0 then repace it with 1. 
    # As the item is not purchased in the last 3 months the unique customer will be 0.
    df_engagement[
        "article_availability_3m"
    ] = df_engagement.article_availability_3m.apply(lambda x: 1 if x == 0 else x)
    df_engagement[
        "article_availability_lt"
    ] = df_engagement.article_availability_lt.apply(lambda x: 1 if x == 0 else x)

    df_engagement["engagement_lt"] = (
        df_engagement.u_customers_lt * df_engagement.t_purchases_lt
    ) / df_engagement.article_availability_lt
    df_engagement["engagement_3m"] = (
        df_engagement.u_customers_3m * df_engagement.t_purchases_3m
    ) / df_engagement.article_availability_3m

    ## both enagegement have very skewed distribution
    ## recent engagement of a product will determine the overall effect
    ## high value indicate good generic engagement
    ## 75th Percentile - Also known as the third, or upper, quartile.
    ## The 75th percentile is the value at which 25% of the answers lie 
    ## above that value and 75% of the answers lie below that value.

    df_engagement["engagement_score"] = np.sqrt(
        df_engagement.engagement_lt * df_engagement.engagement_3m
    )

    # print(df_engagement.engagement_score.min(), df_engagement.engagement_score.max())

    ## if we look at a small section of unengaged items to generate negative samples
    ## then lower quantile value would make sense
    quantile_limit = df_engagement.engagement_score.quantile(quantile)
    # print(df_engagement.engagement_score.size, quantile_limit)

    low_engagement_products = df_engagement.article_id[
        df_engagement.engagement_score < quantile_limit
    ].unique()
    # print("number of products to sample", low_engagement_products.size)

    return low_engagement_products


def get_agg_results(interacted_items_val, products_train, products_val):
    """ """
    collection_dict = {}
    for row in tqdm(interacted_items_val.itertuples()):
        collection_dict.update(
            {row.Index: get_prod_candidates(row, products_train, products_val)}
        )

    aggregate_results_ = pd.DataFrame.from_dict(
        collection_dict, orient="index"
    ).reset_index()

    del collection_dict

    aggregate_results_.columns = [
        "customer_id",
        "avail_products_train",
        "avail_products_val",
    ]

    # aggregate_results_.to_parquet(write_filepath.parent/'temp_storage'/'agg_results.parquet')

    return aggregate_results_


def get_interacted_items(df, df_type="train"):
    """ """
    result = df.groupby(["customer_id"]).parallel_apply(lambda x: x.article_id.unique())

    col_name = "article_id_tr" if df_type == "train" else "article_id_val"

    result = pd.DataFrame(result, columns=[col_name]).reset_index()

    return result


def leave_one_out_ds(
    train,
    val,
    write_filepath,
    n_neg,
    strategy,
    path_temp_storage=None,
    train_article_tx_features=None,
    val_article_tx_features=None,
    quantile=None,
):
    """ """
    if strategy == 2:
        products_train = pd.read_parquet(path_temp_storage / train_article_tx_features)
        products_train = compute_engagement(products_train, quantile=quantile)

        products_val = pd.read_parquet(path_temp_storage / val_article_tx_features)
        products_val = compute_engagement(products_train, quantile=quantile)
    else:
        # find unique products in each dataset
        products_train = np.sort(train.article_id.unique())
        products_val = np.sort(val.article_id.unique())

    # find unique customer in each dataset
    custs_train = np.sort(train.customer_id.unique())
    custs_val = np.sort(val.customer_id.unique())

    # pandas series with customer id as index and interacted product as an array
    print("----  " + "aggregate train")
    interacted_items_train = get_interacted_items(train)

    print("----  " + "aggregate validation")
    interacted_items_val = get_interacted_items(val, df_type="val")

    gc.collect()

    # In CF all customers in val should be in train
    # preparing validation dataset
    interacted_items_val = interacted_items_val.merge(
        interacted_items_train, on="customer_id", how="inner"
    ).set_index(["customer_id"])

    print("----  " + "val: generate product candidates")
    # results = interacted_items_val.parallel_apply(
    #     lambda x: get_prod_candidates(x, products_train, products_val), axis=1
    # )
    # res = pd.DataFrame(results, columns=["article_id_samples"]).reset_index()
    # temp_df_ = pd.DataFrame(
    #     res.article_id_samples.to_list(),
    #     columns=["avail_products_train", "avail_products_val"],
    # )
    # res = res.filter(["customer_id"]).merge(temp_df_, left_index=True, right_index=True)

    results_ = get_agg_results(interacted_items_val, products_train, products_val)
    gc.collect()

    print("----  " + "val: generate negative samples")

    val_neg, train_common_neg = gen_negative_examples(results_, n_neg)

    print("----  " + "train common: generate negative samples")

    assert (
        (train_common_neg.groupby(["customer_id", "purchase"]).article_id.nunique())
        != n_neg
    ).sum() == 0
    assert (
        (val_neg.groupby(["customer_id", "purchase"]).article_id.nunique()) != n_neg
    ).sum() == 0

    gc.collect()

    mask_common_custs = np.isin(custs_train, custs_val)
    train_only_cust = custs_train[~mask_common_custs]
    interacted_items_train = interacted_items_train[
        interacted_items_train.customer_id.isin(train_only_cust)
    ]

    print("----  " + "train only: generate negative samples")
    train_only_dict = {}
    for row in tqdm(interacted_items_train.itertuples()):
        train_only_dict.update(
            {
                row.customer_id: [
                    products_train[~np.isin(products_train, row.article_id_tr)]
                ]
            }
        )

    train_only_df = pd.DataFrame.from_dict(
        train_only_dict, orient="index"
    ).reset_index()
    train_only_df.columns = ["customer_id", "article_id"]

    train_only_cust_neg = gen_negative_examples(
        interacted_items_train, "article_id_tr", n_neg, False
    )

    assert (
        (train_only_cust_neg.groupby(["customer_id", "purchase"]).article_id.nunique())
        != n_neg
    ).sum() == 0

    train_neg = pd.concat([train_only_cust_neg, train_common_neg])

    assert train_neg.customer_id.nunique() == custs_train.shape[0]
    assert (
        (train_neg.groupby(["customer_id", "purchase"]).article_id.nunique()) != n_neg
    ).sum() == 0

    final_train_ = pd.concat([train_neg, train])
    final_val_ = pd.concat([val_neg, val])

    del (train_neg, train, val_neg, val)

    final_train_.to_parquet(write_filepath / "train_ds_stg_{}.parquet".format(strategy))
    final_val_.to_parquet(write_filepath / "val_ds_stg_{}.parquet".format(strategy))
