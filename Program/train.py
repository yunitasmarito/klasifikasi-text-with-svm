#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import MI
import TFIDF
import svm


# In[25]:


def tranning(data,select_count):
    data = pd.read_csv(data, encoding = 'ISO-8859-1')
    X= data.kasus
    y = data.Label
    res, features = MI.selection_feature(select_count,X,y)
    tf_mat_unigram, idf_mat_unigram, tfidf_mat_unigram, terms_unigram,tfidf = TFIDF.generate_tfidf_mat(features, X)
    x_train,x_test,y_train,y_test=train_test_split(tfidf_mat_unigram,y,test_size=0.2, random_state=0)
    with open('D:/coding/model/tfidf','wb') as f:
            pickle.dump(tfidf,f)
            
    SVM = svm.train_svm(x_train,y_train)

    with open('D:/coding/model/Model','wb') as f:
            pickle.dump(SVM,f)
    pred = SVM.predict(x_test)
    akurasi = SVM.score(x_test, y_test)

    return akurasi


# In[ ]:




