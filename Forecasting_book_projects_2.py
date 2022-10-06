# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:48:03 2022

@author: engra
"""


import warnings
from math import sqrt


import pandas as pd  # Basic library for all of our dataset operations



from matplotlib import pyplot as plt

from sklearn.metrics import make_scorer, mean_squared_error




from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox



from utils.metrics import evaluate
# Extra settings
# create a differenced time series
warnings.filterwarnings("ignore")

series = pd.read_csv('datasets/robberies.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('datasets/dataset_robberies.csv', header=False)
validation.to_csv('datasets/validation_robberies.csv', header=False)
series = pd.read_csv('datasets/dataset_robberies.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
 # predict
    yhat = history[-1]
    predictions.append(yhat)
 # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
print(series.describe())
series.plot()
plt.show()
plt.figure(1)
plt.subplot(211)
series.hist()
plt.subplot(212)
series.plot(kind='kde')
plt.show()
groups = series['1966':'1973'].groupby(pd.Grouper(freq='Y'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
plt.show()

def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return pd.Series(diff)

# difference data
stationary = difference(X)
stationary.index = series.index[1:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('datasets/dataset_robberies_stationary.csv', header=False)

plt.figure()
plt.subplot(211)
plot_acf(series, lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(series, lags=50, ax=plt.gca())
plt.show()

#walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# =============================================================================
# 
# 
# def evaluate_arima_model(X, arima_order):
# # prepare training dataset
#   X = X.astype('float32')
#   train_size = int(len(X) * 0.50)
#   train, test = X[0:train_size], X[train_size:]
#   history = [x for x in train]
# # make predictions
#   predictions = list()
#   for t in range(len(test)):
#     model = ARIMA(history, order=arima_order)
#     model_fit = model.fit()
#     yhat = model_fit.forecast()[0]
#     predictions.append(yhat)
#     history.append(test[t])
#   # calculate out of sample error
#   rmse = sqrt(mean_squared_error(test, predictions))
#   return rmse
# 
# 
# def evaluate_models(dataset, p_values, d_values, q_values):
#     dataset = dataset.astype('float32')
#     best_score, best_cfg = float("inf"), None
#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 order = (p,d,q)
#                 try:
#                     rmse = evaluate_arima_model(dataset, order)
#                     if rmse < best_score:
#                         best_score, best_cfg = rmse, order
#                     print('ARIMA%s RMSE=%.3f' % (order,rmse))
#                 except:
#                     continue
#     print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
#     
# # evaluate parameters
# p_values = range(0,13)
# d_values = range(0, 4)
# q_values = range(0, 3)
# evaluate_models(series.values, p_values, d_values, q_values)
# =============================================================================
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = pd.DataFrame(residuals)
plt.figure()
plt.subplot(211)
residuals.hist(ax=plt.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=plt.gca())
plt.show()

X = series.values
transformed, lam = boxcox(X)
print('Lambda: %f' % lam)
plt.figure(1)
# line plot
plt.subplot(311)
plt.plot(transformed)
# histogram
plt.subplot(312)
plt.hist(transformed)
# q-q plot
plt.subplot(313)
qqplot(transformed, line='r', ax=plt.gca())
plt.show()


from math import log
from math import exp
def boxcox_inverse(value, lam):
 if lam == 0:
     return exp(value)
 return exp(log(lam * value + 1) / lam)

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
# transform
     transformed, lam = boxcox(history)
     if lam < -5:
      transformed, lam = history, 1
# predict
     model = ARIMA(transformed, order=(1,1,1))
     model_fit = model.fit()
     yhat = model_fit.forecast()[0]
     # invert transformed prediction
     yhat = boxcox_inverse(yhat, lam)
     predictions.append(yhat)
# observation
     obs = test[i]
     history.append(obs)
     print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
y = validation.values.astype('float32')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
# transform
    transformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
# predict
    model = ARIMA(transformed, order=(0,1,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
# invert transformed prediction
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()