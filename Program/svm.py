#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import LinearSVC


# In[2]:


from sklearn.svm import LinearSVC
def train_svm(X_train, y_train):
    SVM=LinearSVC(C=2)
    SVM.fit(X_train, y_train)
    return SVM

# In[ ]:




