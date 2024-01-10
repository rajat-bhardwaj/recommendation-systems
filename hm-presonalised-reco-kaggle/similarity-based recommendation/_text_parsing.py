import string
import nltk
import string
from nltk.corpus import wordnet
import pandas as pd
import numpy as np


stop_words = nltk.corpus.stopwords.words("english")
s_punct = set(string.punctuation)

word_token = nltk.tokenize.word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()
pos_ = nltk.pos_tag


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.ADV


def cleanText(text):
    """ """
    # tokenise
    text_example = word_token(text.lower())

    ## removing stop words
    text_example = [word for word in text_example if word not in stop_words]
    
    ## remove punctuation as I want to keep words with hypens etc. for e.g. slim-fit, 8-12y
    text_example = [word for word in text_example if word not in s_punct]
    
    ## removing digits
    # text_example = [words for words in text_example if not words.isdigit()]

    ## running POS
    text_example = pos_(text_example)
    
    ## mapping pos to wordnet vocab using pos_tagger
    text_example = list(map(lambda x: (x[0], pos_tagger(x[1])), text_example))
    
    ## running lemmatizer
    text_example = [lemmatizer.lemmatize(word, pos) for word, pos in text_example]
    
    text_example = ' '.join(text_example)
    
    return text_example

def preptext(df_):
    """ """
    ## clean prod description
    results = {}
    for row in tqdm(df_.itertuples()):
        results.update(
            {
                row.article_id: cleanText(
                    row.detail_desc
                )
            }
        )

    temp_dataframe = (
        pd.DataFrame.from_dict(results, orient="index")
        .reset_index()
        .rename(columns={"index": "article_id", 0: 'cleaned_desc'})
        .astype({"article_id": "int32"})
    )
    temp_dataframe = temp_dataframe.merge(df_, on="article_id")

    return temp_dataframe

def get_embeddings(filepath):
    df_ = pd.read_parquet(filepath)
    df_ = df_.set_index(['article_id']).apply(lambda x: x.values, axis=1)
    article_ids = df_.index.values
    values = np.vstack(df_.values)
    return article_ids, values
