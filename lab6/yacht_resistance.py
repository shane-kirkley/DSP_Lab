from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import csv

def get_yacht_data(filename):
    f = open(filename, 'r')
    data_reader = csv.reader(f, dialect='excel')
    a = next(data_reader) # toss labels
    
    S = np.zeros((0,6), np.float)
    y = np.zeros(0, np.float)
    row = np.zeros(5)

    for i, line in enumerate(data_reader):
        row = np.array(line[:6], np.float)
        y = np.hstack((y, np.float(line[6])))
        S = np.vstack((S, row))

    return (S, y)

# Explained Sum of Squares (ESS)
def ess(y, yhat):
    return np.sum((yhat - np.mean(y))**2)

# Total Sum of Squares (TSS)
def tss(y):
    return np.sum((y - np.mean(y))**2)

# Residual Sum of Squares (RSS)
def rss(y, yhat):
    return np.sum((y - yhat)**2)

# R^2 metric
def r_squared(y, yhat):
    return 1 - (rss(y, yhat)/tss(y))

S_all, y = get_yacht_data('yacht_data.csv')
reg = LinearRegression().fit(S_all, y)
print(reg.coef_)

yhat = reg.predict(S_all)

print(r_squared(y, yhat))

plt.scatter(S_all[:,5], y, color='blue')
plt.scatter(S_all[:,5], yhat, color='red')
plt.show()