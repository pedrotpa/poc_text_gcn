import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
from tqdm import tqdm
import logging
from math import factorial


from utils import save_as_pickle, load_pickle



logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def word_word_edges(p_ij):
    word_word=[]
    cols = np.array(list(map(str,p_ij.columns)))
    m = p_ij.to_numpy() > 0
    for i in range(len(cols)-1):
        w1=cols[i]
        word_word.extend([(w1,w2,{"weight":p_ij.loc[w1,w2]}) for w2 in (cols[i+1:])[m[i,i+1:]]])         #cols[m[i]][i:]])
    return word_word

def calculate_tfidf(df_data):
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=lambda x: x, preprocessor=lambda x: x)
    vectorizer.fit(df_data)
    df_tfidf = vectorizer.transform(df_data)
    df_tfidf = df_tfidf.toarray().astype('float32')
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    return pd.DataFrame(df_tfidf,columns=vocab)

def calculate_cooccurrence(df_data, names, window = 10):
    W_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict((name, index) for index, name in enumerate(names))
    W_ij = np.zeros((len(names), len(names)), dtype = np.int32)
    no_windows = 0

    logger.info("Calculating co-occurences...")
    for l in tqdm(df_data, total = len(df_data)):
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])

            for w in d:
                W_i[w] += 1
            for w1,w2 in combinations(d,2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                W_ij[i1][i2] += 1
    W_ij = W_ij.T + W_ij
    
    W_ij = (pd.DataFrame(W_ij, index = names, columns = names)/no_windows).astype('float32')
    W_i = (pd.Series(W_i, index=W_i.keys())/no_windows).astype('float32')
    return  (W_ij, W_i)

def calculate_pmi(p_ij, p_i):
    logger.info("Calculating PMI*...")
    p_ij = p_ij/p_i
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
    p_ij = p_ij + 1E-9
    p_ij = p_ij.apply(np.log)
    return p_ij

def calculate_token_matrix(p_ij, df_tfidf, window = 10):
    logger.info("Calculating Adjacency matrix...")
    len_adj_word = p_ij.shape[0]
    A_size = len_adj_word+len(df_tfidf.index)
    A = np.zeros((A_size,A_size),dtype='float32')
    tmp = p_ij.to_numpy()
    tmp = np.nan_to_num(tmp)
    tmp[tmp<0] = 0
    A[-len_adj_word:, -len_adj_word:] = tmp

    doc_size,_ = df_tfidf.shape
    A[:doc_size, doc_size:]=df_tfidf.to_numpy()
    A[doc_size:, :doc_size] = np.transpose(df_tfidf.to_numpy())

    np.fill_diagonal(A, 1)
    degree = (np.sum(A>0,axis=0)).tolist()
    return A, degree, df_tfidf.index.to_list()+df_tfidf.columns.tolist()

def generate_token_graph(df_data, weird=False,window=10):
    """ generates graph based on text corpus; window = sliding window size to calculate point-wise mutual information between words """    
    
    ### Tfidf
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=lambda x: x, preprocessor=lambda x: x)
    vectorizer.fit(df_data["x"])
    df_tfidf = vectorizer.transform(df_data["x"])
    df_tfidf = df_tfidf.toarray().astype('float32')
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
    
    ### PMI between words
    names = vocab
    n_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict( (name,index) for index,name in enumerate(names) )

    occurrences = np.zeros( (len(names),len(names)) ,dtype=np.int32)
    # Find the co-occurrences:
    no_windows = 0; logger.info("Calculating co-occurences...")
    for l in tqdm(df_data["x"], total=len(df_data["x"])):
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])

            for w in d:
                n_i[w] += 1
            for w1,w2 in combinations(d,2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1
    del df_data

    logger.info("Calculating PMI*...")
    ### convert to PMI
    p_ij = (pd.DataFrame(occurrences, index = names,columns=names)/no_windows).astype('float32')
    p_i = (pd.Series(n_i, index=n_i.keys())/no_windows).astype('float32')

    del occurrences
    del n_i
    p_ij = p_ij/p_i
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
    p_ij = p_ij + 1E-9
    p_ij = p_ij.apply(np.log)

    len_adj_word = p_ij.shape[0]
    A_size = len_adj_word+len(df_tfidf.index)
    A = np.zeros((A_size,A_size),dtype='float32')
    tmp = p_ij.to_numpy()
    tmp = np.nan_to_num(tmp)
    tmp[tmp<0] = 0
    A[-len_adj_word:, -len_adj_word:] = tmp

    doc_size,_ = df_tfidf.shape
    A[:doc_size, doc_size:]=df_tfidf.to_numpy()
    A[doc_size:, :doc_size] = np.transpose(df_tfidf.to_numpy())

    np.fill_diagonal(A, 1)
    if weird:
        degree = [df_tfidf.shape[1]]*df_tfidf.shape[0] + (np.sum(p_ij.to_numpy()>0,axis=0)+np.array([df_tfidf.shape[0]]*df_tfidf.shape[1])).tolist()
    else:
        degree = (np.sum(A>0,axis=0)).tolist()
    return A, degree, df_tfidf.index.to_list()+names.tolist()