#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


my_data = pd.read_csv("drug.csv")
my_data.head(10)


# In[ ]:


my_data.shape # returns (rows,columns)


# In[ ]:


X = my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
X[0:5]


# In[ ]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) # X[:,1]=all rows second column = [F,M,M,F,F]

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])
X[0:5]


# In[ ]:


y = my_data["Drug"]
y[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
x_trainset, x_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[ ]:


print(x_trainset.shape)
print(x_testset.shape)


# In[ ]:


print(y_trainset.shape)
print(y_testset.shape)


# In[ ]:


drugTree=DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf = 5)
drugTree


# In[ ]:


drugTree.fit(x_trainset,y_trainset)


# In[ ]:


predTree = drugTree.predict(x_testset)


# In[ ]:


print(predTree [0:5])
print(y_testset [0:5])


# In[ ]:


from sklearn import metrics
print("Decisiontree's accuracy:",metrics.accuracy_score(y_testset, predTree))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn import tree
results = confusion_matrix(y_testset, predTree)
print('Confusion Matrix :')
print(results)


# In[ ]:


#random forest
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


my_data1 = pd.read_csv('drug.csv')
my_data1[0:5]


# In[ ]:


my_data1.shape


# In[ ]:


X = my_data1[['Age', 'Sex','BP','Cholesterol','Na_to_K']].values
X[0:5]


# In[ ]:


d = pd.get_dummies( my_data1['Sex'])
d.head()


# In[ ]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])
X[0:5]


# In[ ]:


y = my_data["Drug"]
y[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
x_trainset1, x_testset1, y_trainset1, y_testset1 = train_test_split(X, y, test_size=0.3, random_state=3)


# In[ ]:


print(x_trainset1.shape)
print(x_testset1.shape)


# In[ ]:


print(y_trainset1.shape)
print(y_testset1.shape)


# In[ ]:


classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', min_samples_leaf = 5, random_state=0)
classifier.fit(x_trainset1, y_trainset1)


# In[ ]:


y_pred1 = classifier.predict(x_testset1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_testset1, y_pred1)
print(cm)


# In[ ]:




