#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def main():
    file=r"bayes_dataset(1).csv"
    DATA=[]
    fd=csv.reader(open(file))
    for line in fd:
        print(line)
        DATA.append(line)


    data=DATA[0:]
    yescount=0
    nocount=0
    for line in data:
        if line[-1]=='yes':
            yescount+=1
            print(yescount)
        else:
            nocount+=1
    n=len(data)
    pyes=yescount/n
    pno=nocount/n
    print("ENter the car features like color,type and origin")
    x,y,z=input().split()
    pxyes,pxno=bayes(data,x,0,yescount,nocount)
    pyyes,pyno=bayes(data,y,1,yescount,nocount)
    pzyes,pzno=bayes(data,z,2,yescount,nocount)
    resyes=pyes*pxyes*pyyes*pzyes
    resno=pno*pxno*pyno*pzno
    percentageyes=(resyes/(resyes+resno))*100
    percentageno=(resno/(resyes+resno))*100
    p=[percentageyes,percentageno]
    label=['car_stolen_%','car_not_stolen_%']
    plt.pie(p,labels=label)
    plt.show()
    print("percentage yes =",percentageyes,"percentage no =",percentageno)
    def bayes(data,x,col,yescount,nocount):
      xyes=0
      xno=0
      for line in data:
          if line[col]==x:
              if line[-1]=='yes':
                  xyes+=1
          else:
              xno+=1
      pxyes=xyes/yescount
      pxno=xno/nocount
      return pxyes,pxno

main()


# In[ ]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bayes(data, x, col, yescount, nocount):
    xyes = 0
    xno = 0
    for line in data:
        if line[col] == x:
            if line[-1] == 'yes':
                xyes += 1
        if line[col] == x:
            if line[-1] == 'no':
                xno += 1
    pxyes = xyes / yescount
    pxno = xno / nocount
    return pxyes, pxno

def main():
    file = r"bayes_dataset(1).csv"
    DATA = []
    fd = csv.reader(open(file))
    for line in fd:
        print(line)
        DATA.append(line)

    data = DATA[1:]  # Skip header row
    yescount = 0
    nocount = 0
    for line in data:
        if line[-1] == 'yes':
            yescount += 1
            print(yescount)
        else:
            nocount += 1
    n = len(data)
    pyes = yescount / n
    pno = nocount / n
    print("Enter the car features like color, type, and origin:")
    x, y, z = input().split()
    pxyes, pxno = bayes(data, x, 0, yescount, nocount)
    pyyes, pyno = bayes(data, y, 1, yescount, nocount)
    pzyes, pzno = bayes(data, z, 2, yescount, nocount)
    resyes = pyes * pxyes * pyyes * pzyes
    resno = pno * pxno * pyno * pzno
    percentageyes = (resyes / (resyes + resno)) * 100
    percentageno = (resno / (resyes + resno)) * 100
    p = [percentageyes, percentageno]
    label = ['car_stolen_%', 'car_not_stolen_%']
    plt.pie(p, labels=label)
    plt.show()
    print("Percentage yes =", percentageyes, "Percentage no =", percentageno)

if __name__ == "__main__":
    main()

