# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter, y_parameter


# Function for Fitting our data to Linear model
def linear_model_main(x_parameters, y_parameters, p_value):
    # Create linear regression object
    regression = linear_model.LinearRegression()
    regression.fit(x_parameters, y_parameters)
    predict_outcome = regression.predict(p_value)
    predictions = dict()
    predictions['intercept'] = regression.intercept_
    predictions['coefficient'] = regression.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions


# Function to show the results of linear fit model
def show_linear_line(X_parameters, Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters, Y_parameters, color='blue')
    plt.plot(X_parameters, regr.predict(X_parameters), color='red', linewidth=4)
    plt.show()


X, Y = get_data('input_data.csv')
print X
print Y
predict_value = 700
result = linear_model_main(X, Y, predict_value)
print "Intercept value ", result['intercept']
print "coefficient", result['coefficient']
print "Predicted value: ", result['predicted_value']

show_linear_line(X, Y)
