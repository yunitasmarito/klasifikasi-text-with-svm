#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize



def generate_tfidf_mat(features, X_train):
    cvect = CountVectorizer(vocabulary=features, ngram_range=(1,1))
    counts = cvect.fit_transform(X_train)

    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer( vocabulary=features, ngram_range=(1,1), smooth_idf=False)
    tfs = tfidf.fit_transform(X_train)

    tfidf_mat = normalized_counts.multiply(tfidf.idf_).toarray()
    
    TF = normalized_counts.toarray()
    IDF = tfidf.idf_
    TF_IDF = tfidf_mat
    return TF, IDF, TF_IDF, tfidf.get_feature_names(),tfidf




