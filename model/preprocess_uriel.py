# %%
import re
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Union, Mapping, Any

# %%
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from nltk.corpus import wordnet
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# %%
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import datetime
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# %%
#data
#from scraper import get_data
#df=get_data()
#df = pd.read_csv("C:\\Users\\uriel\\OneDrive\\Desktop\\myproject\\local Uriel\\myproject\\new_try.ipynb\\spokesperson.csv")
#df.head()

# %%
#df.to_csv(r"C:\\Users\\uriel\\OneDrive\\Desktop\\myproject\\local Uriel\\myproject\\new_try.ipynb\\spokesperson.csv")
#df = pd.read_csv("spokesperson.csv")  
'''
remove # when inserting a file
'''

# %%
def is_question(words):
    try:
        frst2wrds = f'{words[0].lower()} {words[1].lower()}'
        first6words = ' '.join(words[0:6])
        if words[0].lower() == 'a:':
            return False

        elif words[0].lower() == 'q:':
            return True

        elif 'wang' in frst2wrds or 'zhao' in frst2wrds or 'chunying' in frst2wrds:
            return False

        elif ':' in first6words:
            return True
            
        else:
            return False
    except:
        pass

# %%
def qa_sort(df):
    dialog_no = 0
    data = []
    for row in df.itertuples():
        words = row[3].split()
        result = is_question(words)
        if result == True:
                dialog_no +=1
        data.append([row[3] , result, dialog_no])
    qa_df = pd.DataFrame(data , columns=['text' , 'question' , 'dialog_id'])
    return qa_df

# %%
'''
qa_df = qa_sort(df)
df['question'] = qa_df['question']
df['dialog_id'] = qa_df['dialog_id']

'''
#remove ''' when applying function

# %%
'''for i in range(len(df)):
    df['date'][i] = datetime.datetime.strptime(df['date'][i], '%Y-%m-%d')
type(df['date'].iloc[1])'''

# %%
# documents = df[['text']]
'''
cut_date = '2021-09-30'
cut_date_datetime = datetime.datetime.strptime(cut_date, '%Y-%m-%d')
documents = df[["date","text"]][~(df['date'] < cut_date_datetime)]
documents["index"] = range(len(documents['text']))
documents.head()

#documents = df[['text']].head(200)


#print(len(documents))

'''

# %%
##sentences preprocess
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()




# %%
# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are actually english but biased toward Chinese
    return detect_language(s) in {'English','Chinese'}


# %%
#### word level preprocess 


# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed

# %%
# stemming
# p_stemmer = PorterStemmer()


# def f_stem(w_list):
#     """
#     :param w_list: word list to be processed
#     :return: w_list with stemming
#     """
#     return [p_stemmer.stem(word) for word in w_list]


# %%
# filtering out stop words
# create English stop words list
'''
stop_words = (list(
    set(get_stop_words('en'))
    |set(get_stop_words('es'))
    |set(get_stop_words('de'))
    |set(get_stop_words('it'))
    |set(get_stop_words('ca'))
    |set(get_stop_words('pt'))
    |set(get_stop_words('pl'))
    |set(get_stop_words('da'))
    |set(get_stop_words('ru'))
    |set(get_stop_words('sv'))
    |set(get_stop_words('sk'))
    |set(get_stop_words('nl'))
))
'''
def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_typo(w_list)
    #w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list


