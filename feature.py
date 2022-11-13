from flask import Flask,render_template,url_for,request
import pickle
import nltk
import sklearn
from nltk import word_tokenize
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
import scipy
import pickle
import joblib
import gensim
import smart_open
from gensim.models import Word2Vec
import gensim.downloader as api
from string import punctuation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import scipy
from tqdm import tqdm_notebook #for parallel processing
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from thefuzz import fuzz
#%matplotlib inline

model = api.load('fasttext-wiki-news-subwords-300')

# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

pos_tag = nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()               
    
    return x


def word2vec(s):
    words = str(s).lower() #lower the sentence
    words = word_tokenize(words) #tokenize the sentence
    words = [w for w in words if not w in stopwords.words('english')] #Filter out the stop words
    M = []
    for w in words:#loop through each word in the sentence
        try:
            M.append(model[w])#Gensim model converts each word in the sentence to a dimensional vector space and appends to a list.
        except:
            continue
    M = np.array(M) #convert the list to array. Shape = (num_of_words_in_sentence,300)
    v = M.sum(axis=0) #Sum up along the num_of_words_in_sentence through 300-dim. Shape = (1,300) 
    return v / np.sqrt((v ** 2).sum()) #normalize the values with the sum

def feature_preparation(ms,sr):

    print('inside feature preparation')
    a = {'Marking Scheme': ms, 'Student Response': sr}
    df = pd.DataFrame(a,columns = ['Marking Scheme','Student Response'],index=[0])
    print('created dataframe')
    #df = feature_preparation(df)


    df['Marking Scheme'] = df['Marking Scheme'].apply(preprocess)
    df['Student Response'] = df['Student Response'].apply(preprocess)

    df['Marking Scheme'] = df['Marking Scheme'].apply(word_tokenize)
    df['Student Response'] = df['Student Response'].apply(word_tokenize)
 
    ms_vectors = np.zeros(df.shape[0], 150, dtype='uint8') #vector matrix of shape = (num_of_rows,300) for Marking Scheme
    for i, q in enumerate(tqdm_notebook(df["Marking Scheme"].values)):
        ms_vectors[i, :] = word2vec(q) #function call for each Marking Scheme
        
    sr_vectors  = np.zeros(df.shape[0], 150, dtype='uint8') #vector matrix of shape = (num_of_rows,300) for Student Response
    for i, q in enumerate(tqdm_notebook(df["Student Response"].values)):
        sr_vectors[i, :] = word2vec(q) #function call for each Student Response
        
    print('question vectors done')

    df['len_ms'] = df["Marking Scheme"].apply(lambda x: len(str(x)))
    df['len_sr'] = df["Student Response"].apply(lambda x: len(str(x)))
    df['common_words'] = df.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split()).intersection(set(str(x['Student Response']).lower().split()))), axis=1)
    df['words_total'] = df.apply(lambda x: len(set(str(x['Marking   Scheme']).lower().split()).union(set(str(x['Student Response']).lower().split()))), axis=1)
    df['word_share'] = df['common_words']/df['words_total']
    df['ms_n_words'] = df['Marking Scheme'].apply(lambda row : len(str(row).split(' ')))
    df['st_n_words'] = df['Student Response'].apply(lambda row : len(str(row).split(' ')))
    df['fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))]
    df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))]
    df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))]
    df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))]
    df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))]
    df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(ms_vectors), np.nan_to_num(sr_vectors))] 

    df.drop(['Marking Scheme', 'Student Response'], axis = 1, inplace = True)
    return df