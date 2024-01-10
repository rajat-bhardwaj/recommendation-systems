import sys
sys.path.append("../../")

import faiss
import pandas as pd
import numpy as np
import _text_parsing as txt_
from Data_pipelines import _read_data_files_helper as helper


def setup_prod_embeddings_lookup(path_, nlist=50):
    """
    Reads embeddings from file and generates a FAISS index
    
    PARAMETERS
    ----------
    path_: Posix filepath
    
    RETURNS
    -------
    FAISS index
    
    """
    article_ids, _embeddings = txt_.get_embeddings(path_)
    
    quantizer = faiss.IndexFlatL2(_embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, _embeddings.shape[1], nlist)

    index.train(_embeddings)
    index.add(_embeddings)

    print("total number of available products ", index.ntotal)
    
    return index

def get_top_k_items(fiass_index, query, k=10):
    _, _index = fiass_index.search(query, k)
    return _index

def get_customer_data(path_meta_feat, path_tx_feat) -> (np.array, np.array):
    """ 
    
    PARAMETERS
    ----------
    path_meta_feat: Posix filepath
    path_tx_feat: Posix filepath
    
    RETRUNS
    -------
    numpy array, numpy array
    """
    
    cust_meta_feat = helper.read_customer_file(path_meta_feat).drop(columns=['postal_code'])
    cust_tx_feat = pd.read_parquet(path_tx_feat)

    customer_df = cust_tx_feat.merge(cust_meta_feat, on='customer_id', how='left')
    
    if customer_df.club_member_status.isna().sum() == 0:
        customer_df['club_member_status'] = customer_df.club_member_status.fillna('unknown')
    if customer_df.fashion_news_frequency.isna().sum() == 0:
        customer_df['fashion_news_frequency'] = customer_df.fashion_news_frequency.fillna('unknown')

    customer_df['club_member_status'] = customer_df.club_member_status.str.lower().factorize()[0]
    customer_df['fashion_news_frequency'] = customer_df.fashion_news_frequency.str.lower().factorize()[0]
    
    customer_ids = customer_df.customer_id
    
    customer_df.set_index(['customer_id'], inplace=True)

    cust_embeddings = customer_df.values
    
    return customer_ids, cust_embeddings

def setup_cust_embeddings_lookup(cust_embeddings, nlist=50):

    quantizer = faiss.IndexFlatL2(cust_embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, cust_embeddings.shape[1], nlist)

    index.train(cust_embeddings)
    index.add(cust_embeddings)

    print("total number of customers ", index.ntotal)
    
    return index
