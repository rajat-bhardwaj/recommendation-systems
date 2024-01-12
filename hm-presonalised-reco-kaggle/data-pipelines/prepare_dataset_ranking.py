# ## Prepare dataset
#
# - Candidate Selection (Retrieval): Matrix Factorisation; Generate `n` negative per positive example. (Weekly)
# - Ranking: (Hybrid); New derived features for both customer and products.
# - Development set: Training and Validation; train from `2018 week-38` to `2020 Week-32`
# - Test set: From `2020 Week-33` to `2020 Week-39`

# ### Read datasets

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandarallel import pandarallel

import _read_data_files_helper as data_files

warnings.filterwarnings("ignore")
# setup pandarallel
pandarallel.initialize(nb_workers=7, progress_bar=True, verbose=2, use_memory_fs=False)


# ### Ranking:
#
# Generate customer and product features from transaction table
#
# Development set
# - train set: 2018 week-38 to 2020 Week-28 (max date: 2020-07-12)
# - validation set: 2020 Week-29 to 2020 Week-32 (4 weeks of validation data)
# - i.e. train model on historic data of two years and evaluate performance on upcoming week(s)
#
# Features
# - for each customer and product with their last active date in the train set,
# - Compute features over 3 months, 6-months, one year and lifetime period.

# #### Compute max date in each dataset


def get_last_active_date(input_df, feature, df_name) -> pd.DataFrame:
    """
    Calculates the last active date of given feature
    and merge it with original dataframe

    Parameters
    ----------
    input_df : Pandas DataFrame
    feature : String
        Feature name to run groupby
    df_name: String
        Name of the dataset (Train, Validation or Test)

    Returns
    -------
    Pandas DataFrame
    """
    col_name = df_name + "_" + feature + "_max_dt"
    results = (
        input_df.groupby(feature, as_index=False)
        .agg({"t_dat": "max"})
        .rename(columns={"t_dat": col_name})
    )

    return results


def run_max_date(row, path_base, df_transactions):
    """
    Parameters
    ----------
    input_df : Pandas DataFrame
    feature : String
        Feature name to run groupby
    df_name: String
        Name of the dataset (Train, Validation or Test)

    Returns
    -------
    Pandas DataFrame
    """

    df_name = row.df_name.iloc[0]
    feature = row.feature.iloc[0]

    train_df, val_df, test_df = data_files.get_model_dfs(df_transactions)

    if df_name == "train":
        data_ = train_df
    elif df_name == "val":
        data_ = val_df
    elif df_name == "test":
        data_ = test_df

    # print("#Customers: " + df_name , data_.customer_id.nunique())
    # print("#Articles:  " + df_name, data_.article_id.nunique())

    df_ = get_last_active_date(data_, feature, df_name)

    assert df_[feature].nunique() == data_[feature].nunique()

    return df_


def get_dates_df(path_base, path_model_data):
    """ """
    dummy_max_date_df = pd.DataFrame(
        [
            (df_name, feat)
            for df_name in ["train", "val", "test"]
            for feat in ["customer_id", "article_id"]
        ],
        columns=["df_name", "feature"],
    )

    ## adding only age column as only age feature is used.
    ## If there are more maual features one can join both customer and product features
    df_customers = data_files.read_customer_file(path_base)
    df_transactions = data_files.read_transactions(path_base)
    df_combined = df_transactions.merge(
        df_customers.filter(["customer_id", "age"]), on="customer_id"
    )

    org_rows = df_combined.shape[0]
    org_columns = df_combined.shape[1]

    # run parallel run for max date computation

    res_ = dummy_max_date_df.groupby(["df_name", "feature"]).parallel_apply(
        lambda x: run_max_date(x, path_base, df_transactions.copy())
    )

    for i, row in enumerate(dummy_max_date_df.itertuples()):
        df_name = row.df_name
        feature = row.feature
        df_ = (
            res_.loc[(df_name, feature)]
            .dropna(axis=1, how="any")
            .astype({feature: "int64"})
        )

        df_combined = df_combined.merge(df_, on=feature, how="left")

        ## there will be null values for customers not in other datasets
        ## for example customers outside training dataset will have null for train_customer_max_dt)
        # assert df_combined.isna().sum().sum() == 0

        assert df_combined.shape[0] == org_rows
        assert df_combined.shape[1] == org_columns + i + 1

    ## there will be missing values in the newly created date columns
    df_combined.to_parquet(path_model_data / "combined_dataframe.parquet", index=False)


# #### Generate features
#
# - Train, validate and test dataset are partitioned by time.
# - Features are split over time as follows lifetime, 12 months, 6 months, 3 months for both customer and product.
# - To run parallel execution, each process will
#     - read the original transaction file
#     - filter dataframe for each duration ( 4 partitions )
#     - compute fratures for each subset of data
#     - process the output to store as a parquet file
def get_combined_data(path_model_data, sample=False) -> pd.DataFrame:
    """
    Reads data from a parquet file

    Parameters
    ----------
    path_model_data : str
        filepath to store model input data
    sample : bool, optional
        A flag to generate a small sample for testing

    Returns
    -------
    Pandas DataFrame
    """

    df_ = (
        pd.read_parquet(
            path_model_data / "combined_dataframe.parquet",
        )
        .astype(
            {
                "customer_id": "int64",
                "article_id": "int64",
                "tx_year": "category",
                "tx_month": "int8",
                "tx_week": "int8",
                "price": "float32",
                "sales_channel_id": "category",
                "age": "float32",
            }
        )
    )

    return df_


def _features_custs(df) -> list:
    """
    generate customer features based on transaction sample

    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    list of features
    """
    n_sales_channel = df.sales_channel_id.nunique()
    t_price = df.price.sum()
    u_articles = df.article_id.nunique()
    t_transactions = df.tx_month.count()
    u_acive_days = df.t_dat.nunique()

    return [n_sales_channel, t_price, u_articles, t_transactions, u_acive_days]


def _features_articles(df) -> list:
    """
    Generates article features based on transaction sample
    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    list
    """
    last_purchase_date = df.t_dat.max()
    first_purchase_date = df.t_dat.min()
    u_customers = df.customer_id.nunique()
    t_purchases = df.price.count()
    latest_price = df[df.t_dat == last_purchase_date].price.max()
    discount = df.price.max() - df.price.min()
    article_availability = (last_purchase_date - first_purchase_date).days
    median_age_buyers = df.age.median()

    return [
        u_customers,
        t_purchases,
        latest_price,
        discount,
        article_availability,
        median_age_buyers,
    ]


def get_df(input_, feature_max_date, months):
    """
    Filter dataset based on the time period
    """
    min_date = input_[feature_max_date].iloc[0] - pd.DateOffset(months=months)
    max_date = input_[feature_max_date].iloc[0]

    filtered_ = input_[input_["t_dat"].between(min_date, max_date)]
    return filtered_


def compute_features(x, feature_max_date, agg_feature_name) -> list:
    """
    Compute features for customer / product over following period
    36: 3 years previous / lifetime
    12: 12 months previous
    6: 6 months previous
    3: 3 months previous

    Parameters
    ----------
    x : Pandas dataframe
        groupby input dataframe for each customer/product
    feature_max_date : String
        date feature to select as a max date (for each dataset train, val and test)
    agg_feature_name: String
        Feature to use foor aggregation / groupby

    Returns
    -------
    list
        list of computed features
    """

    duration_list = [36, 12, 6, 3]

    dfs_ = [get_df(x, feature_max_date, months) for months in duration_list]

    if agg_feature_name == "customer_id":
        features = [_features_custs(df_) for df_ in dfs_]
    elif agg_feature_name == "article_id":
        features = [_features_articles(df_) for df_ in dfs_]

    return features


def get_timebased_features(row, path_model_data, sample=False) -> pd.DataFrame:
    """
    Reads the dataset from parquet file and applied feature computation
    (this function is run in each process)

    Parameters
    ----------
    dummy_df : Pandas dataframe
        Dummy dataset created to support parallel execution
    path_model_data : String
        path where model input parquet is stored
    sample : bool, optional
        A flag to generate a small sample for testing

    Returns
    -------
    Pandas DataFrame

    """

    input_df = get_combined_data(path_model_data, sample=sample)
    # remove null values for customer or articles not in the given dataset
    input_df = input_df[~input_df[row.date_feature].isna()]

    df_aggregates = input_df.groupby([row.agg_feature]).parallel_apply(
        lambda x: compute_features(x, row.date_feature, row.agg_feature)
    )
    gc.collect()

    return df_aggregates


def transform_features_df(x, path_model_data, col_features_cust, col_features_articles):
    """
    Transforms the dataset obtained by parallel execution of feature generation
    Process dataset per row
    creates a new df for group by customer_id & article_id and 'train,val & test'

    Parameters
    ----------
    x : Pandas Series
    col_features_cust : list
        column names for customer df
    col_features_articles : list
        column name for article df

    Returns
    -------
    None

    ARGS

    """

    data_type = (
        {"customer_id": "int64"}
        if x.agg_feature.iloc[0] == "customer_id"
        else {"article_id": "int64"}
    )

    ## since the group by is applied in parallel we can extract the only row
    results_ = x.features.iloc[0].to_frame().rename(columns={0: "features"})
    results_flatten = results_.features.apply(lambda x: np.ravel(x))
    results_flatten = pd.DataFrame(
        results_flatten.to_list(),
        columns=col_features_cust
        if x.agg_feature.iloc[0] == "customer_id"
        else col_features_articles,
        index=results_.index,
    ).reset_index()

    filename_ = (
        x.dataset_type.iloc[0]
        + "_"
        + x.agg_feature.iloc[0].split("_")[0]
        + "_tx_features.parquet"
    )

    (
        results_flatten.astype(data_type).to_parquet(
            path_model_data / filename_, index=False
        )
    )


def run_feature_computation(dummy_df, path_model_data, sample=False):
    """
    Run feature generation pipeline
    1. Fix feature column name for both customer and article
    2. parallel execution of time partitioned feature generation
    3. parallel execution of feature dataset transformation

    Parameters
    ----------
    dummy_df : Pandas DataFrame
        Dummy dataset created to support parallel execution
    path_model_data: String
        path where model input parquet is stored

    Returns
    -------
    None
    """

    # set colum names
    col_features_cust = [
        "n_sales_channel",
        "t_amt_spend",
        "u_articles",
        "t_transactions",
        "u_acive_days",
    ]
    col_features_articles = [
        "u_customers",
        "t_purchases",
        "latest_price",
        "discount",
        "article_availability",
        "median_age_buyers",
    ]
    col_names = ["lt", "12m", "6m", "3m"]

    col_features_cust = [
        col + "_" + value for value in col_names for col in col_features_cust
    ]
    col_features_articles = [
        col + "_" + value for value in col_names for col in col_features_articles
    ]

    ## run parallel execution of feature generation
    results_ = []
    for row in dummy_df.itertuples():
        print(row.agg_feature + "------------" + row.date_feature)
        results_.append(get_timebased_features(row, path_model_data, sample=sample))

    ## rehape input and write to file
    temp_results_ = dummy_df.drop(columns=["index"])
    temp_results_["dataset_type"] = temp_results_.date_feature.apply(
        lambda x: str(x).split("_")[0]
    )
    temp_results_["features"] = results_

    _ = temp_results_.groupby(["agg_feature", "date_feature"]).parallel_apply(
        lambda x: transform_features_df(
            x, path_model_data, col_features_cust, col_features_articles
        )
    )


# ### Test feature generation
def test_(row, sample_data_):
    """
    Generates dataset based on the time period and computes original features

    Parameters
    ----------
    row : Pandas row (iterator)

    sample_data_: Pandas dataframe
        Dataframe subset for single customer or product

    Returns
    ---------
    Features: dict
        key value pair of item id (customer or article) and generated features
    """

    def _foo(row, months, input_):
        min_date = input_[row.date_feature].min() - pd.DateOffset(months=months)
        max_date = input_[row.date_feature].min()

        filtered_ = input_[input_["t_dat"].between(min_date, max_date)]
        return filtered_

    duration_list = [36, 12, 6, 3]
    dfs_ = [_foo(row, months, sample_data_) for months in duration_list]

    if row.agg_feature == "customer_id":
        features = [_features_custs(df_) for df_ in dfs_]
        features = {sample_data_.customer_id.min(): features}

    elif row.agg_feature == "article_id":
        features = [_features_articles(df_) for df_ in dfs_]
        features = {sample_data_.article_id.min(): features}

    return features


def test_example(row, path_model_data, sample_data_):
    """
    1. Reads combined dataframe
    2. Take a random sample

    """

    ## column names here matches the order of feature generation
    ## col names also matches the order in duration
    ## there fore the col_features_cust or col_features_articles will generate all features for single duration

    col_names = ["lt", "12m", "6m", "3m"]
    col_features_cust = [
        "n_sales_channel",
        "t_amt_spend",
        "u_articles",
        "t_transactions",
        "u_acive_days",
    ]
    col_features_articles = [
        "u_customers",
        "t_purchases",
        "latest_price",
        "discount",
        "article_availability",
        "median_age_buyers",
    ]
    col_features_cust = [
        col + "_" + value for value in col_names for col in col_features_cust
    ]
    col_features_articles = [
        col + "_" + value for value in col_names for col in col_features_articles
    ]

    result = test_(row, sample_data_)
    df_ = [
        pd.DataFrame(
            np.ravel(value).reshape(1, -1),
            columns=col_features_cust
            if row.agg_feature == "customer_id"
            else col_features_articles,
            index=[key],
        )
        for key, value in result.items()
    ][0]

    return df_


# #### Run comparion
def run_comparision(row, dummy_df, path_model_data, test_data):
    """ """
    filename_ = (
        str(row.date_feature).split("_")[0]
        + "_"
        + row.agg_feature.split("_")[0]
        + "_tx_features.parquet"
    )
    filepath = path_model_data / filename_
    data_type = (
        {"customer_id": "int64"}
        if row.agg_feature == "customer_id"
        else {"article_id": "int64"}
    )

    print(row.agg_feature + "  ------  " + row.date_feature)
    computed_df = pd.read_parquet(filepath).astype(data_type)

    if row.agg_feature == "customer_id":
        test_item = computed_df.customer_id.sample(1).values[0]
        sample_data_ = test_data[test_data.customer_id.isin([test_item])].copy()
        org_df = computed_df[computed_df.customer_id == test_item].set_index(
            ["customer_id"]
        )

    else:
        test_item = computed_df.article_id.sample(1).values[0]
        sample_data_ = test_data[test_data.article_id.isin([test_item])].copy()
        org_df = computed_df[computed_df.article_id == test_item].set_index(
            ["article_id"]
        )

    test_df = test_example(row, path_model_data, sample_data_)

    try:
        assert test_df.compare(org_df).sum().sum() == 0
    except AssertionError:
        return test_df, org_df


if __name__ == "__main__":
    path_base = Path("")
    path_model_data = Path("")

    dummy_df = pd.DataFrame(
        [
            ["customer_id", "train_customer_id_max_dt"],
            ["customer_id", "val_customer_id_max_dt"],
            ["customer_id", "test_customer_id_max_dt"],
            ["article_id", "train_article_id_max_dt"],
            ["article_id", "val_article_id_max_dt"],
            ["article_id", "test_article_id_max_dt"],
        ],
        columns=["agg_feature", "date_feature"],
    ).reset_index()

    _ = get_dates_df(path_base, path_model_data)
    gc.collect()

    _ = run_feature_computation(dummy_df, path_model_data)
    gc.collect()

    test_data = get_combined_data(path_model_data)
    for i, row in enumerate(dummy_df.itertuples()):
        run_comparision(row, dummy_df, path_model_data, test_data)
