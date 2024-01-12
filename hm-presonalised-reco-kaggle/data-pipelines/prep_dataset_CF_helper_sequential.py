import gc
import os
import warnings
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

avail_cpus = multiprocessing.cpu_count() - 1

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
pandarallel.initialize(nb_workers=avail_cpus, progress_bar=True, verbose=2, use_memory_fs=False)

warnings.filterwarnings("ignore")
tqdm.pandas(desc="Negative samples!")

""" 
For both train and val set, (with positive interactions)
1. Find product pool (unique items)
2. Find customer pool (unique users)
3. Aggregate interacted items for customers.

Since this is CF, the val set should have customers available in train set.
*** Generate product candidates based on `leave-one-out`.
These customers are common in train and val set, so we can generate their negative
samples simultaneously from train and val product pool.

4. Generate product candidate from train and val pool for common customers.
5. For remaining customers in train (not available in val), generate negative samples from train prod pool.
6. Merge train positive dataset with train only negative and common customers.
7. Merge val positive dataset with val negative dataset.
8. write to file.

"""


def _get_random_(x, n_neg, replace=False):
    return np.random.choice(x, n_neg, replace=replace)


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

    # if article_availability_3m is 0 then repace it with 1. As the item is not purchased in the last 3 months the unique customer will be 0.
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
    ## The 75th percentile is the value at which 25% of the answers lie above that value and 75% of the answers lie below that value.

    df_engagement["engagement_score"] = np.sqrt(
        df_engagement.engagement_lt * df_engagement.engagement_3m
    )

    ## if we look at a small section of unengaged items to generate negative samples
    ## then lower quantile value would make sense
    quantile_limit = df_engagement.engagement_score.quantile(quantile)

    low_engagement_products = np.sort(
        df_engagement.article_id[df_engagement.engagement_score < quantile_limit].unique())

    return low_engagement_products


def get_agg_results(df_in_array, products_train, products_val) -> pd.DataFrame:
    """ 
    Generates product canddates for common users both in training and validation set
    
    Parameters
    ----------
    df_in_array: numpy array
        Numpy array with pandas dataframe object
    products_train: numpy array
        Products available in train set
    products_val: numpy array
        Products available in train set
    Returns
    -------
    Pandas Dataframe
    """

    def _run_prod_candidates(df_):
        collection_dict = {}
        for row in tqdm(df_.itertuples()):
            collection_dict.update(
                {row.Index: get_prod_candidates(row, products_train, products_val)}
            )

        aggregate_results_ = pd.DataFrame.from_dict(
            collection_dict, orient="index"
        ).reset_index()

        collection_dict.clear()
        del collection_dict

        aggregate_results_.columns = [
            "customer_id",
            "avail_products_train",
            "avail_products_val",
        ]

        return aggregate_results_

    output = pd.concat([_run_prod_candidates(df) for df in df_in_array])

    gc.collect()

    return output


def get_interacted_items(df, df_type="train"):
    """ """
    collec = []

    groups_ = df.groupby(["customer_id"])
    for group in tqdm(groups_):
        collec.append([group[0], group[1].article_id.unique()])

    col_name = "article_id_tr" if df_type == "train" else "article_id_val"

    results_ = pd.DataFrame(collec, columns=["customer_id", col_name])

    collec.clear()
    del collec
    gc.collect()

    return results_


def __setup_df(_df, feature_name):
    _df = _df.explode(feature_name)
    _df["purchase"] = 0
    return _df


def gen_negative_examples(new_df, n_neg):
    new_df["article_id_tr"] = new_df.avail_products_train.parallel_apply(
        lambda x: _get_random_(x, n_neg)
    )
    new_df["article_id_val"] = new_df.avail_products_val.parallel_apply(
        lambda x: _get_random_(x, n_neg)
    )

    train_df = __setup_df(
        new_df.filter(["customer_id", "article_id_tr"]), "article_id_tr"
    )
    val_df = __setup_df(
        new_df.filter(["customer_id", "article_id_val"]), "article_id_val"
    )

    train_df.rename(columns={"article_id_tr": "article_id"}, inplace=True)
    val_df.rename(columns={"article_id_val": "article_id"}, inplace=True)

    new_df = None
    del new_df

    return train_df, val_df


def gen_negative_examples_train(input_df, products_train, n_neg):
    def _train_sample(df_sampple):
        train_only_dict = {}
        for row in tqdm(df_sampple.itertuples()):
            train_only_dict.update(
                {
                    row.customer_id: [
                        _get_random_(
                            products_train[~np.isin(products_train, row.article_id_tr)],
                            n_neg,
                        )
                    ]
                }
            )

        train_only_df = pd.DataFrame.from_dict(
            train_only_dict, orient="index"
        ).reset_index()
        train_only_df.columns = ["customer_id", "article_id"]

        train_only_dict.clear()
        del train_only_dict

        #
        train_only_df = __setup_df(train_only_df, "article_id")

        return train_only_df

    output = pd.concat([_train_sample(df) for df in input_df])
    gc.collect()

    return output


def leave_one_out_ds_sequence(
    train,
    val,
    write_filepath,
    n_neg,
    strategy,
    path_features_data=None,
    train_article_tx_features=None,
    val_article_tx_features=None,
    quantile=None,
    n_splits=10,
):
    """ """
    if strategy == 2:
        products_train = pd.read_parquet(path_features_data / train_article_tx_features)
        products_train = compute_engagement(products_train, quantile=quantile)

        products_val = pd.read_parquet(path_features_data / val_article_tx_features)
        products_val = compute_engagement(products_val, quantile=quantile)
    else:
        products_train = np.sort(train.article_id.unique())
        products_val = np.sort(val.article_id.unique())

    custs_train = np.sort(train.customer_id.unique())
    custs_val = np.sort(val.customer_id.unique())
    
    # products in val should be available in train
    # sampling negatives from this ensures this policy
    mask_val_ = np.isin(products_val, products_train, assume_unique=True)
    products_val = products_val[mask_val_]

    # pandas series with customer id as index and interacted product as an array
    print("----  " + "aggregate products for train set")
    interacted_items_train = get_interacted_items(train)

    print("----  " + "aggregate products for val set")
    interacted_items_val = get_interacted_items(val, df_type="val")

    # In CF all customers in val should be in train
    # preparing validation dataset
    # creating df with aggregated products for both train and val set
    interacted_items_val = interacted_items_val.merge(
        interacted_items_train, on="customer_id", how="inner"
    ).set_index(["customer_id"])
    
    print("----  " + "val set: generate product candidates")
    ## split datframe into multiple subsets -- memory constraints
    df_array = np.array_split(interacted_items_val, n_splits)

    results_ = get_agg_results(df_array, products_train, products_val)

    df_array = None
    del df_array, mask_val_

    print("----  " + "val & train common custs: generate negative samples")
    train_common_neg, val_neg = gen_negative_examples(results_, n_neg)

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

    print("----  " + "train only custs: generate negative samples")
    df_train_only = np.array_split(interacted_items_train, n_splits)
    train_only_cust_neg = gen_negative_examples_train(
        df_train_only, products_train, n_neg
    )

    assert (
        (train_only_cust_neg.groupby(["customer_id", "purchase"]).article_id.nunique())
        != n_neg
    ).sum() == 0

    train_neg = pd.concat([train_only_cust_neg, train_common_neg])

    del train_only_cust_neg, train_common_neg

    assert train_neg.customer_id.nunique() == custs_train.shape[0]
    assert (
        (train_neg.groupby(["customer_id", "purchase"]).article_id.nunique()) != n_neg
    ).sum() == 0

    final_train_ = pd.concat([train_neg, train])

    # val positive dataset has customers not present in train set
    # remove these cusotmers for CF
    ## removing products from positive validation set that are not in train
    val = val[val.customer_id.isin(val_neg.customer_id.unique())]
    val = val[val.article_id.isin(products_train)]
    
    final_val_ = pd.concat([val_neg, val])

    del (train_neg, train, val_neg, val)
    gc.collect()

    final_train_.to_parquet(write_filepath / "train_ds_stg_{}.parquet".format(strategy))
    final_val_.to_parquet(write_filepath / "val_ds_stg_{}.parquet".format(strategy))


##### NEGATIVE EXAMPLE FOR 
def neg_ex(df_, write_filepath, n_neg, replace=False):
    """ 
    Generates negative examples on weekly dataset.
    The groupby columns can be used to generate datasets for 
    monthly or other frequency.
    """
    # numpy array
    
    weekly_dataset = df_.filter(['customer_id', 'article_id', 'purchase']).drop_duplicates()
    filename = str(df_.tx_year.iloc[0]) + '_' + str(
            df_.tx_week.iloc[0]) + '_.parquet'
    
    item_all = weekly_dataset['article_id'].unique()
    interact_status = weekly_dataset.groupby(['customer_id'
                                  ]).agg({'article_id': 'unique'})
    interact_status['article_id'] = interact_status['article_id'].apply(
        lambda x: item_all[~np.isin(item_all, x)])
    interact_status['article_id'] = interact_status['article_id'].apply(
        lambda x: np.random.choice(x, n_neg, replace=replace))

    interact_status = interact_status.explode('article_id')
    interact_status['purchase'] = 0
    df_final = pd.concat([weekly_dataset, interact_status.reset_index()])

    del (interact_status)
    
    # same file name as input but different folder
    df_final.to_parquet(write_filepath / filename)

    del (df_final)


def generate_negative_examples(path_base, write_filepath, n_neg):
    """ 
    Generates negative examples for each positive 
    example in the dataset by using all avaiable products

    PARAMETERS:
    -----------
    filepath: PosixPath
        Path of transaction data csv file split weekly
    write_filepath: String
        Path to store model data after processing
    """

    df_transactions = data_files.read_transactions(path_base)
    df_transactions['purchase'] = 1

    (df_transactions.groupby(
        ['tx_year',
         'tx_week']).parallel_apply(lambda x: neg_ex(x, write_filepath, n_neg)))
    
    gc.collect()
