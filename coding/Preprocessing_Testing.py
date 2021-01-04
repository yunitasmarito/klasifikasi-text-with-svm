#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



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


def Preprocessing(kasus):
    cl = clean(kasus)
    cf = casefolding(cl)
    stop = remove_stop_words(cf)
    stem = stemming(stop)
    tok = token(stem)
    kasus = tok
    kasus=[str(kasus)]
    return kasus

