#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer


def selection_feature(select_count, X_train, y_train):
    cv = CountVectorizer(max_features=15000)
    X_vec = cv.fit_transform(X_train)
    res = sorted(list(zip(cv.get_feature_names(), mutual_info_classif(X_vec, y_train, discrete_features=True))), key=lambda x: x[1], reverse=True)[0:select_count]
    selected_features = list(set([x[0] for x in res]))
    return res, selected_features


