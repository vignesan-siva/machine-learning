import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\linear regresssion\\short_size.csv")
x=data.iloc[:,[0,1]].values#independent values
y=data.iloc[:,[2]].values#dependent values
#split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
#fitting the traing data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)