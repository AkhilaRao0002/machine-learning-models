#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
a = []
print("\n The Given Training data Set")
with open('enjoysport.csv','r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)
num_attributes = len(a[0])-1


# In[2]:


print("\n The initial value of hypothesis:")
S = ['0'] * num_attributes
G = ['?'] * num_attributes
print("\n The most Specific hypothesis S0 : [0,0,0,0,0,0]")
print("\n The most General hypothesis G0 : [?,?,?,?,?,?]")


# In[3]:


for j in range(0,num_attributes):
    S[j]=a[0][j];


# In[4]:


S


# In[13]:


print("\n Candidate Elimination algorithm Hypothesis Version Space computation\n")
temp=[]
for i in range(0,len(a)):#0 to 24
    if a[i][num_attributes]=='Yes':# if last row is yes
        for j in range(0,num_attributes):#0 to 6 = 0,1,2,3,4,5{for one row}
            if a[i][j]!=S[j]:# not matching with s
                S[j]='?'
        for j in range(0,num_attributes):#0 to 6
            for k in range(1,len(temp)):#if temp is zero no execution
              if temp[k][j]!='?' and temp[k][j]!=S[j]:
                  del temp[k]
        #print(len(temp))
        print("--------------------------------------------------------------")
        print("For training example no:{0} the hypothesis is S{0} ".format(i+1),S)
        if (len(temp)==0):
            print("For training example no :{0} the hypothesis is G{0}".format(i+1),G)
        else:
            print("For positive training example no :{0} the hypothesis is G{0}".format(i+1),G)
        a[i][num_attributes]=='No'
        for j in range(0,num_attributes):
             if S[j] !=a[i][j] and S[j]!= '?':# s not ? and s not in a
                 G[j]=S[j]
                 temp.append(G)
                 #print(temp)
                 G = ['?'] * num_attributes
        print("--------------------------------------------------------------")
        print("For training example no:{0} the hypothesis is S{0} ".format(i+1),S)
        print("For training example no:{0} the hypothesis is G{0} ".format(i+1),temp)


# In[ ]:




