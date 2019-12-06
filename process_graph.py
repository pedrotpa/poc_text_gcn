from process_input import *
from utils import *

import numpy as np

def process_graph(df, language='english', window = 10):
    df = tokenize(df, language)
    df_tfidf = calculate_tfidf(df)
    names = df_tfidf.columns
    p_ij = calculate_pmi(*calculate_cooccurrence(df, names, window = 10))
    del df
    A, degrees, node_names = calculate_token_matrix(p_ij, df_tfidf)
    del p_ij
    del df_tfidf
    degrees = np.array(degrees,dtype='float32')**(-1/2)
    A_hat = degrees*A*degrees[:, np.newaxis]
    return A_hat, node_names