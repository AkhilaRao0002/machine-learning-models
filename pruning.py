#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data ='heart(1).csv'
df = pd.read_csv(data)
df.head()
df.info()


# In[4]:


X = df.drop(columns =['target'])
y = df['target']
print(X.shape)
print(y.shape)


# In[5]:


x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(x_train.shape)
print(x_test.shape)


# In[6]:


clf =tree.DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)


# In[15]:


plt.figure(figsize=(9,9))
features = df.columns
classes=['Not heart disesse','heart disease']
tree.plot_tree(clf,feature_names = features,class_names=classes,filled=True)
plt.show()


# In[11]:


def plot_confusionmatrix(y_train_pred,y_train,dom):
    print(f'{dom} Confusion matrix')#f is for converting into string in heatmap
    cf = confusion_matrix(y_train_pred,y_train)
    sns.heatmap(cf,annot=True,yticklabels=classes,xticklabels=classes,cmap='Blues',fmt='g')
    plt.tight_layout()
    plt.show()


# In[10]:





# In[9]:


print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')


# In[13]:


plot_confusionmatrix(y_train_pred,y_train,y_test)

