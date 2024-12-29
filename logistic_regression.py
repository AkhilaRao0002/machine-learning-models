#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


# In[ ]:


data = pd.read_csv('diabetes(2).csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


#checking correlation to use common feature for highly correlated variables
corr = data.corr()
corr


# In[ ]:


sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)


# In[ ]:


dfTrain = data[:650]
dfTest = data[650:750]
dfCheck = data[750:]


# In[ ]:


trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',axis = 1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',axis = 1))


# In[ ]:


means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)

trainData = (trainData - means)/stds
testData = (testData - means)/stds


# In[ ]:


diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData,trainLabel)
accuracy = diabetesCheck.score(testData,testLabel)
print("accuracy = ",accuracy * 100,"%")


# In[ ]:


coeff = list(diabetesCheck.coef_[0])
coeff


# In[ ]:


labels = list(dfTrain.drop('Outcome',axis = 1).columns)
labels


# In[ ]:


features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# In[ ]:




