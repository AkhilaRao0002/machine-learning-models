#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[ 'figure.figsize' ] = (9.0 , 7.0)


# In[ ]:


data = pd.read_csv('data.csv')
X = data.iloc[:,0]
Y = data.iloc[:,1]
plt.scatter(X,Y)
plt.xlabel('height')
plt.ylabel('weight')
plt.title('scatter plot between height and wieght')
plt.show()


# In[ ]:


X_mean = np.mean(X)
Y_mean = np.mean(Y)

num=0
den=0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean

print(data)
print("X_ mean = ")
print(X_mean)
print("Y_ mean = ")
print(Y_mean)
print("slope = ")
print(m)
print("intercept = ")
print(c)


# In[ ]:


Y_pred = m*X+c

plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(Y_pred,Y)
print(MSE)

