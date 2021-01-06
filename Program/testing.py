#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Preprocessing_Testing
import pickle


# In[ ]:


with open('D:/coding/model/Model','rb') as f:
    mp = pickle.load(f)


# In[ ]:


with open('D:/coding/model/tfidf','rb') as f:
     tfi = pickle.load(f)


# In[ ]:


#read data testing
def test(data):
    with open(data) as f:
        kasus = f.read()
    text= Preprocessing_Testing.Preprocessing(kasus)
    tf=tfi.transform(text)
    predictions=mp.predict(tf)
    if predictions == 0:
        predictions ="27 ayat 1"
    elif predictions == 1:
        predictions ="27 ayat 3"
    elif predictions == 2:
        predictions ="27 ayat 4"  
    elif predictions == 3:
        predictions ="28 ayat 1"
    elif predictions == 4:
        predictions ="27 ayat 2"
    else:
        predictions ="Pasal lainya"
    
    return kasus,text,predictions

