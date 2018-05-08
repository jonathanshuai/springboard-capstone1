# Skeleton file for basic exploratory analysis
import os
import copy

import time
from datetime import datetime
import dateutil

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


# import pendulum

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot


feature_file = './data/cleaned/feature.csv'
annual_file = './data/cleaned/annual_gun.csv'
overall_file = './data/cleaned/overall.csv'

feature_df = pd.read_csv(feature_file, parse_dates=True, index_col=0)
annual_df = pd.read_csv(annual_file, parse_dates=True, index_col=0)
overall_df = pd.read_csv(overall_file, parse_dates=True, index_col=0)

def test_stationarity(time_series, window=12):    
    #Determing rolling statistics
    rolling = time_series.rolling(window=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    #Plot rolling statistics:
    orig = plt.plot(time_series, color='blue',label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xticks(rotation=45)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(time_series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)




time_series = feature_df.groupby('next_date').sum()['next_deaths']
time_series.index = pd.to_datetime(time_series.index)

smoothed = np.log(time_series)
smoothed = smoothed - smoothed.rolling(window=6).mean()
smoothed = smoothed - smoothed.shift()
smoothed = smoothed.dropna()
test_stationarity(smoothed)

time_series_log = np.log(time_series)
decomposition = seasonal_decompose(time_series_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(time_series_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

test_stationarity(smoothed)

