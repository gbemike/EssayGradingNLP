from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
#import distance
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import scipy
#import xgboost as xgb
from fuzzywuzzy import fuzz
import pickle
import joblib
#import gensim
import smart_open
#from gensim.models import Word2Vec
from string import punctuation
stop_words = stopwords.words('english')

app = Flask(__name__)

with open('rfBasicFt.pkl', 'rb') as file: #load model as pickle file
    pickle_model = pickle.load(file) 

def feature_preparation(ms,sr):
    print('inside feature preparation')
    a = {'Marking Scheme': ms, 'Student Response': sr}
    testdf = pd.DataFrame(a,columns = ['Marking Scheme','Student Response'],index=[0])
    print('created dataframe')
    
    print('question vectors done')
    
    testdf['len_q1'] = len(ms)
    #testdf.ms.apply(lambda x: len(str(x)))
    testdf['len_q2'] = len(sr)
    #testdf.sr.apply(lambda x: len(str(x)))
    #testdf['ms_n_words'] = testdf.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split())))
    #testdf['st_n_words'] = testdf.apply(lambda x: len(set(str(x['Student Response']).lower().split())))
    testdf['ms_n_words'] = testdf['Marking Scheme'].apply(lambda row : len(str(row).split(' ')))
    testdf['st_n_words'] = testdf['Student Response'].apply(lambda row : len(str(row).split(' ')))
    testdf['common_words'] = testdf.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split()).intersection(set(str(x['Student Response']).lower().split()))), axis=1)
    testdf['words_total'] = testdf.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split()).union(set(str(x['Student Response']).lower().split()))), axis=1)
    testdf['words_share'] = testdf.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split()).intersection(set(str(x['Student Response']).lower().split()))), axis=1)/(testdf.apply(lambda x: len(set(str(x['Marking Scheme']).lower().split()).union(set(str(x['Student Response']).lower().split()))), axis=1))
    testdf['fuzz_ratio'] = testdf.apply(lambda x: fuzz.ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    testdf['fuzz_partial_ratio'] = testdf.apply(lambda x: fuzz.partial_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    testdf['fuzz_partial_token_set_ratio'] = testdf.apply(lambda x: fuzz.partial_token_set_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    testdf['fuzz_partial_token_sort_ratio'] = testdf.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    testdf['fuzz_token_set_ratio'] = testdf.apply(lambda x: fuzz.token_set_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    testdf['fuzz_token_sort_ratio'] = testdf.apply(lambda x: fuzz.token_sort_ratio(str(x['Marking Scheme']), str(x['Student Response'])), axis=1)
    
    testdf.drop(['Marking Scheme','Student Response'],axis = 1, inplace = True)

    return testdf
    
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    print('inside predict')
    if(request.method == 'POST'):
        ms = request.form['marking scheme']
        sr = request.form['student response']
        predictdf = feature_preparation(ms,sr)
        result = pickle_model.predict(predictdf)
        return render_template('result.html',prediction = result[0])
        
        
if __name__ == '__main__':
	app.run()
