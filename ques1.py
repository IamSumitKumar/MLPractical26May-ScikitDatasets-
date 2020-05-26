#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets


# In[2]:


from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
print(X.shape)


# In[4]:


print(X)


# In[5]:


print(y)


# In[6]:


#X = Boston.data[:, :13]  
#Y = Boston.target


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)


# In[9]:


linearMod = LinearRegression()


# In[10]:


linearMod.fit(X_train,y_train)


# In[11]:


y_pred=linearMod.predict(X_test)
y_pred


# In[12]:


R_sq = linearMod.score(X, y)
print('coefficient of determination:', R_sq)


# In[13]:


from sklearn.metrics import r2_score


# In[14]:


Score = r2_score(y_test, y_pred)


# In[15]:


Score


# In[ ]:




