#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


def most_common(lst):
    return max(set(lst), key=lst.count)
def euclidean(point, data): #for computing distance from k to data
    return np.sqrt(np.sum((point - data)**2, axis =1))
class KNeighboursClassifier:
    def __init__(self, k, dist_metric=euclidean):
        self.k=k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbours = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbours.append(y_sorted[:self.k])
        return list(map(most_common, neighbours))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)
accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighboursClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="accuracy",
       title="Performance of knn")
plt.show()


# In[ ]:




