# H&M Personalized Fashion Recommendations
### Provide product recommendations based on previous purchases

This section contains solution based on Kaggle dataset provided by H&M for fashion recommendation. 

## Data

Original dataset and competion can be found [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview).

## Data Anlaysis (EDA)

Check the EDA section for more details.

Things to consider while developing a recommendation solution

- bulk purchases by customers
- very few purchases by customers
- least selling articles
- most selling articles

## Recommendation solution

- Retrieval OR Candidate Selection
- Ranking
- Evaluation

## Data Prep for recommendation problem

`ML problem statement:` Train model on historic data of two years and evaluate performance on upcoming week(s)

- Development set (training and validation) from `2018 week-38` to `2020 Week-32`
- Test set: From `2020 Week-33` to `2020 Week-39`


### Negative sampling (per customer)

- Strategy One: Follow `leave-one-out` policy and select `n` articles at random.
- Strategy Two: Follow `leave-one-out` policy and select `n` articles based on lowest performing criteria.
- Strategy Three: Generate negative samples per week at random.
- Strategy four: Generate negatives based on last `n` transactions per customer.

- Number of negative samples: 
    - 5 for NeuMF
    - 30 total

### Ranking: 

Generate customer and product features from transaction table.

Features
- for each customer and product with their last active date in the train set,
- Compute features over 3 months, 6-months, one year and lifetime period. (example)

For customers:
- `n_sales_channel`: Number of sales channel used.
- `t_price`: Total amount spent.
- `u_articles`: Total number of articles purchased.
- `t_transactions`: Total transactions.
- `u_acive_days`: Total number of days of purchase.

For articles:
- `u_customers`: Number of unique customers made a purchase.
- `t_purchases`: Total number of units sold.
- `latest_price`: Latest price in the dataset/window.
- `discount`: Discount offerened.
- `article_availability`: Number of days the product is offered. (first buy and last)
- `median_age_buyers`: Median age of buyer.

## Retreival

- NeuMF:Neural Matrix Factorisation. (Pytorch)
- Product Similarity (Pytorch)
    - Product metadata.
    - Product features extracted from transactions.
    - Visual embeddings extracted from available images.
    - Text embeddings extracted from product product description.(pytorch implementation)

- `ToDo`: Customer Similarity
- `ToDo`: Items bought together
- `ToDo`: Two Tower (Merlin models)
- `ToDo`: Variational Autoencoder

## Ranking
- `ToDo`: Facebook’s DLRM
    - The dataset can contain multiple categorical features. DLRM requires that all categorical inputs are fed through an embedding layer with the same dimensionality. 
 

### Evaluation
- Average Hit Rate over validation set.
- NDCG

## Results

