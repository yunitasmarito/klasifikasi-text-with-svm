#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Preprocessing_Testing
import pickle


# In[ ]:


with open('C:/Users/ASUS/Desktop/Model','rb') as f:
    mp = pickle.load(f)


# In[ ]:


with open('C:/Users/ASUS/Desktop/tfidf','rb') as f:
     tfi = pickle.load(f)


# In[ ]:


#read data testing
def test(data):
    with open(data) as f:
        kasus = f.read()
    text= Preprocessing_Testing.Preprocessing(kasus)
    tf=tfi.transform(text)
    predictions=mp.predict(tf)
    predik = format(predictions)
    
    return kasus,text,predik

