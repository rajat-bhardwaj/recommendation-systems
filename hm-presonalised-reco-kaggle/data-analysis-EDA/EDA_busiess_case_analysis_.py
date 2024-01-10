import os
import pathlib
import gc
import warnings
from time import time

import numpy as np
import pandas as pd
import dask.dataframe as dd

import data_preprocessing as dp

warnings.filterwarnings("ignore")


class EdaBusinessCase:
    """
    Runs EDA via python class

    """

    def __init__(self, path_base):

        self.frac_sample = 0.01
        self.window_size = 3

        ## read file
        self.path_base = path_base

        self.df_customers = dd.from_pandas(
            dp.read_customer_file(self.path_base), sort=False,
            chunksize=10000).persist(scheduler="processes")
        self.df_articles = dd.from_pandas(
            dp.read_articles(self.path_base), sort=False,
            chunksize=10000).persist(scheduler="processes")
        self.df_transactions = dd.from_pandas(
            dp.read_transactions(self.path_base), sort=False,
            chunksize=1000000).persist(scheduler="processes")

        self.df_article_subset = self.df_articles[[
            "article_id",
            "prod_name",
            "graphical_appearance_name",
            "perceived_colour_value_name",
            "index_group_name",
            "garment_group_name",
        ]].persist(scheduler="processes")

        self.df_joined_tx = self.df_transactions.merge(
            self.df_article_subset, on=["article_id"]).persist(scheduler="processes")

    def daily_aggregate(self):
        """ """
        daily_aggregate = (self.df_transactions.compute(
            scheduler="processes").groupby(["t_dat", "sales_channel_id"]).agg({
                "customer_id": ["nunique", "count"],
                "article_id": "nunique",
                "price": "sum",
            }).reset_index())
        daily_aggregate.columns = [
            "_".join(col) for col in daily_aggregate.columns.values
        ]
        daily_aggregate.rename(
            columns={
                "customer_id_nunique": "no_unique_custs",
                "customer_id_count": "n_transactions",
                "article_id_nunique": "n_unique_articles",
                "price_sum": "amt_purchase",
            },
            inplace=True,
        )

        daily_aggregate["tx_per_cust"] = (daily_aggregate.n_transactions /
                                          daily_aggregate.no_unique_custs)
        daily_aggregate["u_articles_per_cust"] = (
            daily_aggregate.n_unique_articles / daily_aggregate.no_unique_custs)
        daily_aggregate["amt_per_tx"] = (daily_aggregate.amt_purchase /
                                         daily_aggregate.n_transactions)

        return daily_aggregate

    def article_subset_info(self):
        """ """
        article_info = (self.df_article_subset
                .describe(include=["object", "category"])
                .compute(scheduler='threads')
                .T
                .drop(columns="count")
                .sort_values(by="unique", ascending=False))

        return article_info

    def season_analysis(self):

        df_season = (self.df_joined_tx.groupby([
            "tx_year", "article_id"
        ]).t_dat.nunique().compute(scheduler="processes").reset_index())
        return df_season

    def products_per_year(self):
        """ """

        df_prod_per_year = (self.df_joined_tx.groupby(
            ["tx_year"]).prod_name.nunique().compute(
                scheduler="processes").reset_index().rename(
                    columns={"prod_name": "n_unique_prods"}))
        return df_prod_per_year

    def return_available_dates(self):

        return (self.df_transactions.groupby(["tx_year"])["t_dat"].agg(
            ["min", "max"]).compute(scheduler="processes"))

    def prod_season(self):
        """ """
        df_prod_per_year = self.products_per_year()
        
        df_prod_season = (self.df_joined_tx.groupby([
            "tx_year", "prod_name"
        ]).tx_month.nunique().compute(scheduler="processes").reset_index())

        df_prod_season2 = (df_prod_season.groupby(["tx_month", "tx_year"
                                                  ]).count().reset_index())
        df_prod_season2 = df_prod_season2.merge(df_prod_per_year, on="tx_year")
        df_prod_season2["prod_year_prop_sale"] = (
            df_prod_season2.prod_name / df_prod_season2.n_unique_prods)

        return df_prod_season2

    def colour_analysis1(self):
        """ """
        df_color = (self.df_joined_tx.groupby([
            "garment_group_name", "tx_year", "tx_month"
        ]).customer_id.nunique().compute(scheduler="processes").reset_index())
        df_color.columns = [
            "graphical_appearance_name",
            "tx_year",
            "tx_month",
            "n_unique_customers",
        ]  #'n_unique_articles', 'n_unique_products']

        return df_color

    def colour_analysis2(self):
        """ """

        ## perceived_colour_value_name
        df_color = (
            self.df_joined_tx.groupby(
                ["perceived_colour_value_name", "tx_year",
                 "tx_month"])["customer_id"]  # , 'article_id']
            .nunique().compute(scheduler="processes").reset_index())

        df_color.columns = [
            "perceived_colour_value_name",
            "tx_year",
            "tx_month",
            "n_unique_customers",
        ]
        # , 'n_unique_articles', 'n_unique_products', 'n_transactions']

        return df_color

    def colour_analysis3(self):
        """ """

        # perceived color wrt garments
        df_color = (self.df_joined_tx.groupby([
            "perceived_colour_value_name",
            "garment_group_name",
            "tx_year",
        ]).article_id.nunique().compute(scheduler="processes").reset_index())
        df_color.columns = [
            "perceived_colour_value_name",
            "garment_group_name",
            "tx_year",
            "n_unique_articles",
        ]

        return df_color

    def color_analysis4(self):
        """ """
        # perceived color wrt index_group_name
        df_color = (self.df_joined_tx.groupby([
            "garment_group_name", "index_group_name", "tx_year"
        ]).price.sum().compute(scheduler="processes").reset_index())
        df_color.columns = [
            "garment_group_name",
            "index_group_name",
            "tx_year",
            "total_sale_amount",
        ]

        return df_color

    def test_filewrite(self):
        purchase_freq = (self.df_transactions
                 .groupby(["tx_year", "tx_week", "sales_channel_id", "customer_id"],
                                  observed=True,
                                  sort=False)
                 .article_id
                 .count()
                 .persist(scheduler='threads')
                 .to_parquet((self.path_base + 'intermediate_file.parquet'), 
                             compression='gzip', overwrite=True, compute_kwargs={'scheduler': 'processes'}))
                         
        
    def purchase_frequency(self):
        """ """
        a = time()
        purchase_freq = self.df_transactions.compute(scheduler='processes')
        purchase_freq = (purchase_freq
                 .groupby(["tx_year", "tx_week", "sales_channel_id", "customer_id"])
                 .article_id
                 .count()
                 .reset_index())
                         
        purchase_freq = (purchase_freq
                          .groupby(["tx_year", "tx_week", "sales_channel_id", "article_id"])
                          .customer_id
                          .count()
                          .reset_index()
                          .rename(columns={
                    "article_id": "n_transactions",
                    "customer_id": "n_customers"
                }))
        print('execution time ', time() - a)

        return purchase_freq

    def repeating_customer(self):
        """
        # ### Distribution of repeating customers weekly
        # - Lond tail distribution as expected.
        """
        # Somehow converting this to pandas made it work in 45 second but das on Mac WTF
        rep_customer = (self.df_joined_tx
                        .compute(scheduler="processes")
                        .groupby(["tx_year", "sales_channel_id", "customer_id"])
                        .tx_week
                        .nunique()
                        .reset_index())

        return rep_customer

    def purchase_freq(self):
        """ """
        ### number of days a cusotmers has made purchases

        df_purchase_freq = (self.df_transactions
                            .groupby(["tx_year", "customer_id"])
                            .t_dat
                            .count()
                            .compute(scheduler="processes")
                            .reset_index())
        df_purchase_freq = df_purchase_freq[df_purchase_freq.t_dat > 0]
        df_purchase_freq = (df_purchase_freq.groupby(
            ["t_dat", "tx_year"]).customer_id.count().reset_index())

        return df_purchase_freq

    def get_purchase_gap(self, x):
        """ """
        x = x.t_dat.sort_values()
        value = (x - x.shift(periods=1)).astype("timedelta64[D]").fillna(0)
        return value.mean()
        
    def purchase_gap(self):
        """ """
        a = time()
        df_results = (self.df_transactions.groupby([
            "tx_year", "customer_id"
        ]).apply(self.get_purchase_gap).compute(scheduler="processes").reset_index())
        df_results.columns = [
            "tx_year", "customer_id", "average_gap_bw_purchase_days"
        ]
        df_results[
            "average_gap_bw_purchase_days"] = df_results.average_gap_bw_purchase_days.round(
            ).astype(int)
        print("execution time in sec, ", time() - a)

        return df_results

    def get_fraction_data(self):
        """ """
        n_unique = self.df_joined_tx.customer_id.nunique().compute(
            scheduler="processes")
        sample_custs = np.random.choice(
            (self.df_joined_tx.customer_id.unique().compute(
                scheduler="processes")),
            size=int(n_unique * self.frac_sample),
            replace=False,
        )
        print(sample_custs.size)

        return sample_custs

    def bought_together(self):
        """ """
        sample_custs = self.get_fraction_data()

        dask_bought_together = self.df_joined_tx[self.df_joined_tx.customer_id.isin(
            sample_custs)][[
                "prod_name", "customer_id", "t_dat", "garment_group_name"
            ]].persist(scheduler="processes")

        def generate_sets(x):
            prod = x.prod_name.astype(str).unique()
            garment_group = x.garment_group_name.astype(str).unique()

            return prod, garment_group

        a = time()
        dask_bought_together = (dask_bought_together.groupby([
            "customer_id", "t_dat"
        ]).apply(generate_sets).compute(
            scheduler="processes").reset_index().rename(columns={0: "result"}))
        print("execution time in sec, ", time() - a)

        test_df = pd.DataFrame(
            dask_bought_together.result.to_list(),
            columns=["prod_name_together", "garment_group_name_together"],
        )
        test_df = test_df.applymap(lambda x: ", ".join(np.sort(x)))
        test_df = pd.concat([dask_bought_together, test_df],
                            axis=1).drop(columns=["result"])
        test_df = (test_df.groupby([
            "garment_group_name_together"
        ]).customer_id.agg(["nunique", "count"]).reset_index().rename(columns={
            "nunique": "n_unique_custs",
            "count": "total_purchase_days"
        }))

        del dask_bought_together

        test_df = test_df[test_df.n_unique_custs > test_df.n_unique_custs.
                          quantile(0.98, interpolation="nearest")]

        return test_df

    def rebuying_frequency(self):
        """ """
        # ### Customer rebuying frequency
        # - (after first purchase) when's the next purchase on average wrt to garment group etc
        ### running this on pandas using multi processing for experimentation

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window_size)
        sample_custs = self.get_fraction_data()

        test_sample_ = (self.df_joined_tx[self.df_joined_tx.customer_id.isin(
            sample_custs)][[
                "prod_name", "garment_group_name", "customer_id", "t_dat"
            ]].compute(scheduler="processes").set_index("t_dat"))

        test_sample_["prod_name"] = test_sample_.prod_name.astype(str)
        test_sample_[
            "garment_group_name"] = test_sample_.garment_group_name.astype(str)

        df_purchases = dp.generate_next_purchase(test_sample_, indexer)

        df_purchases = df_purchases.rename(
            columns={
                "garment_group_name": "next_garment_group_name",
                "prod_name": "next_prod_name",
            })

        new_df_purchase = test_sample_.reset_index().merge(
            df_purchases, on=["t_dat", "customer_id"])

        new_df_purchase = (new_df_purchase.groupby(
            ["garment_group_name", "next_garment_group_name"]).agg({
                "customer_id": "nunique",
                "t_dat": "count"
            }).reset_index())

        new_df_purchase = new_df_purchase[
            (new_df_purchase.next_garment_group_name != "") &
            (new_df_purchase.t_dat > new_df_purchase.t_dat.quantile(
                0.98, interpolation="nearest"))]

        del test_sample_, df_purchases

        return new_df_purchase
