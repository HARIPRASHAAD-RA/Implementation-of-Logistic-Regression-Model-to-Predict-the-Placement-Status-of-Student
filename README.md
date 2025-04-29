# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARIPRASHAAD RA
RegisterNumber:212223040060
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('pd.csv')
dataset
dataset.info()

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary', axis=1)
dataset.info()

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

dataset.info()

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()

print(x_train.shape)
print(y_train.shape)

from sklearn.linear_model import LogisticRegression
cl=LogisticRegression(max_iter=1000)
cl.fit(x_train,y_train)
y_pred=cl.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
cl.predict([[0,87,0,95,0,2,8,0,0,1,5,6]])
cl.predict([[1,2,3,4,5,6,7,8,9,10,11,12]])


```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


![image](https://github.com/user-attachments/assets/023e8137-d903-4108-b62f-46f596adbaff)
![image](https://github.com/user-attachments/assets/66b397a4-2262-4ffd-ba4b-aad344332cc6)
![image](https://github.com/user-attachments/assets/005fcb7f-1544-4a0b-b6dc-30f17aec2a30)
![image](https://github.com/user-attachments/assets/e44790ea-9cbd-4f59-a5ba-223e173bffb4)
![image](https://github.com/user-attachments/assets/7f698836-4e73-4802-94ea-cf3fa6dda439)
![image](https://github.com/user-attachments/assets/fed7094c-88a0-433c-95a0-285bb72a7826)
![image](https://github.com/user-attachments/assets/6530694c-82a2-4b99-b074-ca3d73982b2d)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
