#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Precossing_Testing 
import pickle


# In[ ]:


with open('desktop/model_pickle','rb') as f:
    mp = pickle.load(f)


# In[ ]:


with open('desktop/tfidf','rb') as f:
     tfi = pickle.load(f)


# In[ ]:


#read data testing
def testing(data):
    with open(data) as f:
        kasus = f.read()
    text= Precossing_Testing.Preprocessing(kasus)
    tf=tfi.transform(text)
    predictions=mp.predict(tf)
    predik = format(predictions)
    
    return kasus, text, predik

