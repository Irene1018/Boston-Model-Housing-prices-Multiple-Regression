#!/usr/bin/env python
# coding: utf-8


#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


#load data
data = pd.read_csv('hw3_data.csv')
data.head(3)

#check corelation of variables
plt.figure(figsize=(10, 10))
cm = np.corrcoef(data.values.T)
sns.set (font_scale = 1.5)
hm = sns.heatmap(cm,
                cbar = True,
                annot = True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size': 10},
                yticklabels = list(data.columns),
                xticklabels = list(data.columns))
plt.show()

#pairplot
sns.pairplot(data, height=2.5)
plt.tight_layout()
plt.show()

#choose x and split train / test
x = data[['RM','LSTAT', 'PTRATIO']]
y = data[['MEDV']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 99)


#train model1
model1 = linear_model.LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
train_pred1 = model1.predict(x_train)

#evaluate the model
MSE = mean_squared_error(y_test, y_pred)
MSE_train1 = mean_squared_error(y_train, train_pred1)
print ("MSE = ", MSE)
print ("MSE_train1 = ", MSE_train1)
score1 = model1.score(x_test, y_test)
print ("Score = ", score1)

#transform x into polynominal features
polynomial_features= PolynomialFeatures(degree=2) 
x_poly = polynomial_features.fit_transform(x)
xp_train, xp_test, yp_train, yp_test =  train_test_split(x_poly, y, test_size = 0.25, random_state = 99)

#train model2
model2 = linear_model.LinearRegression()
model2.fit(xp_train, yp_train)
yp_pred = model2.predict(xp_test)
train_pred = model2.predict(xp_train)

MSE_Train = mean_squared_error(yp_train, train_pred)
MSE = mean_squared_error(yp_test, yp_pred)

print ("MSE = ", MSE)
print ("MSE_Train = ", MSE_Train)

score2 = model2.score(xp_test, yp_test)
print ("Score = ", score2)

#load new data
new_data = pd.read_csv('hw3_prediction.csv')
newx = new_data[['RM','LSTAT', 'PTRATIO']]
newx_poly = polynomial_features.fit_transform(newx)
predict_newy = model2.predict(newx_poly)
#output my answer
my_ans = pd.DataFrame(predict_newy, columns=['new_y'])
my_ans.to_csv('hw3_ans.csv', index=False, header = False)





