# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

8.End the Program
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIJAY K
RegisterNumber:  24901153
*/

import chardet

file ='spam.csv'
with open (file, 'rb' )as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding = 'Windows-1252')
data.head()

data.info()

data.isnull().sum()

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)


```

## Output:
![Uploading image.png…]()

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
