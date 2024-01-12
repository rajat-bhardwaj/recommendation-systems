import sys
import json

import gc

sys.path.append("../")

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

tqdm.pandas(desc="random bar!")

import warnings

warnings.filterwarnings("ignore")

import torch
from tensordict import TensorDict
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import functional as fc

import NeuMF_pytorch_helper as neucf_models

cuda = torch.device("cuda")


def distribute_predictions(
    customer_batch,
    avail_products_torch,
    observed_dict,
    f_params,
    model_,
    n_prods,
    k=100,
):
    """
    Perform evaluation for each batch of customers.

    ## for each customer
    ## split tensor by index
    ## get top k predictions
    ## get article ids for predictions
    ## get ground truth
    ## number of prediction in ground truth

    PARAMETERS
    ----------
    customer_batch: Numpy Array
        Array of customer ids to generate predictions.
    avail_products_torch: Torch Tensor
        Available products from train set.
    n_prods: int
        Total number of available products.
    observed_dict: dict
        Interacted items by customers in the val set.
    model_: Pytorch Lightning model
        Model to use for generating predictions.
    k: int
        Number of recommendations. Default 100.

    Returns
    -------
    sum_hit_rate: int
        Sum of hit rate over all customers in the given batch.
    """

    _dl_config = {
        "batch_size": n_prods,
        "drop_last": False,
        "shuffle": False,
        "num_workers": f_params["predict_num_workers"],
        "pin_memory": f_params["pin_memory"],
        "prefetch_factor": f_params["predict_prefetch_factor"],
        "pin_memory_device": "cuda",
    }

    trainer_pred_ = pl.Trainer(
        accelerator=f_params["accelerator"],
        strategy=f_params["strategy"],
        devices=f_params["devices"],
        enable_model_summary=True,
        inference_mode=True,
        # profiler="pytorch", # ['simple', 'advanced', 'pytorch', 'xla']
    )

    input_cust = (
        (torch.as_tensor(customer_batch)).repeat_interleave(n_prods).reshape(-1, 1)
    )
    prod = avail_products_torch.repeat(customer_batch.shape[0], 1)

    td_ = TensorDataset(input_cust, prod)
    dl = DataLoader(td_, **_dl_config)

    del input_cust, prod

    model_.eval().to(device=cuda)
    # predictions = [(model_(user.to(device=cuda), item.to(device=cuda))) for user, item in tqdm(iter(dl))]

    predictions = trainer_pred_.predict(
        model=model_, dataloaders=dl, return_predictions=True
    )

    observed_dict.to(device=cuda)
    avail_products_torch.to(device=cuda)

    def run_eval_per_cust(customer_id, preds):
        preds.to(device=cuda)
        preds = torch.nn.Sigmoid()(preds)
        _, _indices = torch.topk(preds, k, dim=0)
        top_k_products = avail_products_torch[_indices.flatten()]
        top_k_products.to(device=cuda)
        ground_truth = observed_dict[customer_id]
        ground_truth = ground_truth[ground_truth[0:] > -1]
        hr_cust = torch.sum(
            torch.isin(top_k_products, ground_truth, assume_unique=True)
        ).item()
        return hr_cust

    ## requires batch size to be equal to number of products
    ## processing one customer
    results = [
        run_eval_per_cust(str(customer_batch[i]), preds)
        for i, preds in enumerate(predictions)
    ]

    predictions.clear()
    del td_, dl

    torch.cuda.empty_cache()
    gc.collect()

    return results


def evaluate_(f_params, filepath_val, filepath_train, model, neg_stg, sample=False):
    """
    PARAMETERS:
    -----------
    f_params: dict
        config dict
    filepath_val: Posix filepath
        Filepath for the val set
    filepath_train: Posix filepath
        filepath for the training set
    model_: Pytorch Lightning model
        Model to use for geerating predictions.

    RETURNS:
    ---------
    avg_hit_rate: float
        Average hit rate of the dataset

    """
    # val or test set
    observed_truth = pd.read_parquet(filepath_val)
    # train set
    avail_products = pd.read_parquet(filepath_train)

    if neg_stg == 3:
        customers_ = observed_truth.customer_id.drop_duplicates().sample(
            frac=0.26, replace=False, random_state=42
        )
        observed_truth = observed_truth[observed_truth.customer_id.isin(customers_)]

    if sample:
        customers_ = observed_truth.customer_id.drop_duplicates().sample(
            frac=0.01, replace=False, random_state=42
        )
        observed_truth = observed_truth[observed_truth.customer_id.isin(customers_)]

    # filter datasets
    avail_products = avail_products.article_id.unique()
    observed_truth = observed_truth[observed_truth.article_id.isin(avail_products)]

    avail_prod_torch = torch.as_tensor(avail_products).int().reshape(-1, 1)
    avail_custs = observed_truth.customer_id.unique()

    n_custs = avail_custs.shape[0]
    n_prods = avail_products.shape[0]

    print(" -----  {} customers and {} products".format(n_custs, n_prods))

    interected_items = (
        observed_truth.astype({"customer_id": str})
        .groupby(["customer_id"])
        .article_id.progress_apply(lambda x: x.values)
        .to_frame()
    )
    test_ = (
        pd.DataFrame(interected_items.article_id.to_list()).fillna(-1).astype("int32")
    )
    interected_items = pd.concat(
        [interected_items.reset_index().filter(["customer_id"]), test_], axis=1
    )

    batch_size = [interected_items.shape[1] - 1, 1]

    # converting to torch dict
    # torch dict needs key to be string and values of same size
    # hence filling na with -1
    interected_items = interected_items.set_index(["customer_id"]).progress_apply(
        lambda x: torch.as_tensor(x.values.reshape(-1, 1)), axis=1
    )
    interected_items = interected_items.to_dict()
    intrct_items_torch_dict = TensorDict(
        interected_items,
        batch_size=batch_size,
    )

    interected_items.clear()
    del interected_items

    n_splits = 1 if (n_custs // 500) == 0 else n_custs // 500

    cust_batches = np.array_split(avail_custs, n_splits)
    results_ = []

    for batch in cust_batches:
        results_.extend(
            distribute_predictions(
                batch,
                avail_prod_torch,
                intrct_items_torch_dict,
                f_params,
                model,
                n_prods,
            )
        )

    avg_hit_rate = np.mean(results_)

    results_.clear()

    torch.cuda.empty_cache()
    gc.collect()

    return avg_hit_rate


if __name__ == "__main__":
    start_time = time()
    n_users, n_items = neucf_models.get_cust_item_count(Path(neucf_models.path_data))

    with open("NeuMF_fixed_params.json") as file:
        f_params = json.load(file)

    model_name = "neumf"
    results = {}

    for neg_stg in [1, 2, 3]:
        folder_name = "CF_model_input_neucf_S{}".format(neg_stg)

        ## load the saved model
        model_ = neucf_models.get_pretrained(
            f_params, model_name, neg_stg, n_users, n_items
        )

        filepath_val = Path(
            neucf_models.path_processed_data
        ) / "evaluation_set_stg_{}.parquet".format(neg_stg)
        filepath_train = (
            Path(neucf_models.path_processed_data)
            / folder_name
            / "train_ds_stg_{}.parquet".format(neg_stg)
        )

        hit_rate = evaluate_(
            f_params, filepath_val, filepath_train, model_, neg_stg, sample=False
        )
        print(hit_rate)
        results.update({neg_stg: hit_rate})

    with open("results_ncf.json", "w") as file:
        json.dump(results, file)

    print("total prediction time: {} seconds".format(time() - start_time))
