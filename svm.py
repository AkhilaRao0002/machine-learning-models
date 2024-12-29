#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


#loading iris data
iris = datasets.load_iris()
iris_data_df = pd.DataFrame(iris.data,columns=['sepal length (cm)',
'sepal width (cm)',
'petal length (cm)',
'petal width (cm)'])
iris_target_df=pd.DataFrame(iris.target,columns=['class'])
iris_data_df.head(5)


# In[10]:


# 0=setosa, 1=versicolor,2=virginica
iris_target_df.head(5)


# In[11]:


x = iris.data #features
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 42,test_size=0.2)


# In[12]:


#polynomial kernal
svc_polynomial = SVC(kernel='poly', degree=8, gamma="auto")
svc_polynomial.fit(x_train, y_train) # making prediction
y_pred_polynomial = svc_polynomial.predict(x_test) # evaluating our model
print("Evaluation: Polynomial kernel")
print(classification_report(y_test,y_pred_polynomial))


# In[13]:


#radial kernel
svc_RBF = SVC(kernel='rbf', gamma="auto")
svc_RBF.fit(x_train, y_train) # making prediction
y_pred_RBF = svc_RBF.predict(x_test) # evaluating our model
print("Evaluation:RADIAL kernel")
print(classification_report(y_test,y_pred_RBF))


# In[14]:


#SIGMOID kernel
svc_sig = SVC(kernel='sigmoid', gamma="auto")
svc_sig.fit(x_train, y_train) # making prediction
y_pred_sig = svc_sig.predict(x_test) # evaluating our model
print("Evaluation:sigmoid kernel")
print(classification_report(y_test,y_pred_sig))


# In[15]:


#linear kernel
svc_linear = SVC(kernel='linear', gamma="auto")
svc_linear.fit(x_train, y_train) # making prediction
y_pred_linear = svc_sig.predict(x_test) # evaluating our model
print("Evaluation:linear kernel")
print(classification_report(y_test,y_pred_linear))


# In[16]:


from sklearn.model_selection import GridSearchCV


# In[17]:


grid_parameters = {'C' : [0.1,1,10,100], 'gamma' : [1,0.1,0.01,0.001],'kernel':['rbf','linear','poly','sigmoid']}
grid = GridSearchCV(SVC(),grid_parameters,refit=True,verbose=2)
print(grid.fit(x_train,y_train))


# In[18]:


print(grid.best_estimator_)


# In[19]:


print(sns.heatmap(confusion_matrix(y_test,grid.predict(x_test))))

