import pickle
import nltk

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords):
            tokens1.append(token)
    return tokens1

def tokenize(df_data,language='english'):
    stopwords = list(set(nltk.corpus.stopwords.words(language)))
    return df_data.apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
