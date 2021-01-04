#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import walk
from io import StringIO
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# # **1. Baca Dataset**

# In[2]:


def loaddata():
    dir_name = 'D:/coding/dataset/training' 
    df_list = [] 
    df = pd.DataFrame(columns=['kasus', 'label'])
    for (dirpath, dirnames, filenames) in walk(dir_name):
        for label in dirnames:
            for (dirpath, dirnames, filenames) in walk(dir_name+'/'+label):
                for filename in filenames:
                    f = open(dir_name+'/'+label+'/'+filename, "r")
                    f = StringIO(f.read().replace('\n', ' ') + '\t' + label) #replace new line with tab
                    temp = pd.read_csv(f, sep="\t", names=[ 'kasus', 'label'])
                    df_list.append(temp)
                break
        break
    df = pd.concat(df_list)
    df = df.reset_index()
    df
    return df
loaddata()


# In[3]:


def label():
    data=loaddata()
    col = ['kasus','label']
    data=data[col]
    data= data[pd.notnull(data['kasus'])]

    data.columns=['kasus','label']
    data['Label']= data['label'].factorize()[0]
    return data
dfkasus=label()


# In[4]:


dfkasus


# In[5]:


dfkasus.to_excel('traindata.xlsx')


# # **Preprocessing**
# 

# In[6]:


def clean(kasus):
    kasus = ''.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)|(http\S+)", "", kasus)) #hapus #,@,url
    kasus = re.sub(r'[^A-Za-z\s\/]' ,' ', kasus) #hapus simbol dan tanda baca
    kasus = re.sub(r'_', '', kasus) 
    kasus = re.sub(r'/', ' ', kasus)
    kasus = re.sub(r'\d+', '', kasus) 
    kasus = re.sub(r'\n', ' ', kasus) 
    kasus = re.sub(r'\s{2,}', ' ', kasus)
    return kasus
def remove_stop_words(kasus):
    stop_words = set(stopwords.words('indonesian'))
    stop_words.update(['terdakwa','twitter','akun','status','mengupload','memposting','bbm','facebook','saksi','fb','blackberry','messenger','desember','januari','februari','maret','april','mei','juni','juli','agustus','september','oktober','november','media','sosial','tanggal'])
    no_stop_words=[word for word in kasus.split() if word not in stop_words]
    no_step_sentence = ' '.join(no_stop_words)
    return no_step_sentence

def casefolding(kasus):
    kasus = kasus.lower().strip() #case folding
    return kasus

def stemming(kasus):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(kasus)

def token(kasus):
    tokenizer = RegexpTokenizer('\w+')
    kasus = tokenizer.tokenize(kasus)
    return kasus


hasilpreprocessing = []

for kasus in dfkasus['kasus']:
    cl = clean(kasus)
    cf = casefolding(cl)
    stop = remove_stop_words(cf)
    stem = stemming(stop)
    tok = token(stem)
    hasilpreprocessing.append(tok)
dfkasus['kasus'] = hasilpreprocessing
hasilpreprocessing 


# In[8]:


dfkasus.to_csv('Preprocessing.csv')


# In[ ]:




