
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\linear regresssion\\future_mark_prediction.csv")

x=data.iloc[:,[0]]
y=data.iloc[:,[1]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict([[80],[98]])
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

