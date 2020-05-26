#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


# Write a program using Scikit Learn that utilizes Logistic regression to build a classification
# model using all the four features to predict the class of a plant. Print the confusion matrix,
# accuracy, precision and recall for the model.

# In[2]:


Iris = datasets.load_iris()


# In[3]:


Iris


# In[4]:


print(Iris.data.shape)


# In[5]:


X = Iris.data[:, :4]  # we only take the first two features.
Y = Iris.target


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.25, random_state=42)


# In[8]:


logmodel = LogisticRegression()


# In[9]:


logmodel.fit(X_train,Y_train)


# In[10]:


Y_pred=logmodel.predict(X_test)
Y_pred


# In[11]:


from sklearn import metrics
print("Accuracy of the model ",metrics.accuracy_score(Y_test,Y_pred))


# In[12]:


from sklearn.metrics import confusion_matrix


# In[13]:


confMat = confusion_matrix(Y_test, Y_pred)
confMat


# In[14]:


labels = Y_test
predictions = Y_pred

cm = confusion_matrix(labels, predictions)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


# In[15]:


recall


# In[16]:


precision


# Also, build a classification model in Scikit Learn using Neural Networks using all the features to
# predict the class a plant belongs to. Print the confusion matrix, accuracy, precision and recall for
# the model and compare its performance with the model created using Logistic regression.

# In[17]:


from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# In[18]:


from sklearn.datasets import make_classification


# In[19]:


X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
CLASF = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
CLASF.predict_proba(X_test[:1])

CLASF.predict(X_test[:5, :])

CLASF.score(X_test, y_test)


# In[20]:


labels = Y_test
predictions = Y_pred

cm = confusion_matrix(labels, predictions)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


# In[21]:


cm


# In[22]:


recall


# In[23]:


precision


# In[ ]:




