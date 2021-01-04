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


# In[2]:


df = pd.read_csv("Desktop/Preprocessing.csv", encoding = 'ISO-8859-1')
data=df


# In[3]:


X= data.kasus
y = data.Label


# In[4]:


res, features = MI.selection_feature(3000,X,y)


# In[5]:


tf_mat_unigram, idf_mat_unigram, tfidf_mat_unigram, terms_unigram,tfidf = TFIDF.generate_tfidf_mat(features, X)


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(tfidf_mat_unigram,y,test_size=0.1, random_state=0)


# In[7]:


SVM = svm.train_svm(tfidf_mat_unigram,y)

pred = SVM.predict(x_test)
print(SVM.score(x_test, y_test))
print(confusion_matrix(pred, y_test))
print(classification_report(y_test, pred))


# In[8]:


with open('desktop/model_pickle','wb') as f:
    pickle.dump(SVM,f)


# In[9]:


with open('desktop/tfidf','wb') as f:
    pickle.dump(tfidf,f)


# In[ ]:




