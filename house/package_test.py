# coding=utf-8

import numpy as np
from scipy import special, optimize
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
from sklearn import datasets, linear_model

# scipy test
# f = lambda x: -special.jv(3, x)
# sol = optimize.minimize(f, 1.0)
# x = linspace(0, 10, 5000)
# plt.plot(x, special.jv(3, x), '-', sol.x, -sol.fun, 'o')
# plt.savefig('plot.png', dpi=96)

# numpy test
# a = arange(12)
# a = a.reshape(3,2,2)
# print a

# panda test
# values = np.array([2.0, 1.0, 5.0, 0.97, 3.0, 10.0, 0.0599, 8.0])
# ser = pd.Series(values)
# print ser

# matplotlib test
# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# z = np.cos(x**2)
#
# plt.figure(figsize=(8,4))
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
# plt.plot(x,z,"b--",label="$cos(x^2)$")
# plt.xlabel("Time(s)")
# plt.ylabel("Volt")
# plt.title("PyPlot First Example")
# plt.ylim(-1.2,1.2)
# plt.legend()
# plt.show()

# plt
# example data
# mu = 100  # mean of distribution
# sigma = 15  # standard deviation of distribution
# x = mu + sigma * np.random.randn(10000)
#
# num_bins = 50
# # the histogram of the data
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# # add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)
# plt.plot(bins, y)
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
#
# # Tweak spacing to prevent clipping of y label
# plt.subplots_adjust(left=0.15)
# plt.show()

# ipython test
# def func(x1):
#     return (x1-3)*(x1-5)*(x1-7)+85
#
# a, b = 2, 9  # integral limits
# x = np.linspace(0, 10)
# y = func(x)
#
# fig, ax = plt.subplots()
# plt.plot(x, y, 'r', linewidth=2)
# plt.ylim(ymin=0)
#
# # Make the shaded region
# ix = np.linspace(a, b)
# iy = func(ix)
# verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
# poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
# ax.add_patch(poly)
#
# plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
# horizontalalignment = 'center', fontsize=20)
#
# plt.figtext(0.9, 0.05, '$x$')
# plt.figtext(0.1, 0.9, '$y$')
#
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
#
# ax.set_xticks((a, b))
# ax.set_xticklabels(('$a$', '$b$'))
# ax.set_yticks([])
#
# plt.show()



# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
% np.mean((regr.predict(diabetes_X_test)-diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
