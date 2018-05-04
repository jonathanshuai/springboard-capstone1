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

def show_decomposition(time_series):
    decomposition = seasonal_decompose(time_series)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(time_series, label='Original')
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

    return decomposition 

feature_file = './data/cleaned/feature.csv'
annual_file = './data/cleaned/annual_gun.csv'
overall_file = './data/cleaned/overall.csv'
by_date_total_file = './data/cleaned/by_date_total.csv'

feature_df = pd.read_csv(feature_file, parse_dates=True, index_col=0)
annual_df = pd.read_csv(annual_file, parse_dates=True, index_col=0)
overall_df = pd.read_csv(overall_file, parse_dates=True, index_col=0)
by_date_total_df = pd.read_csv(by_date_total_file, parse_dates=True, index_col=0)



time_series = by_date_total_df.resample('W').sum().sum(axis=1)
time_series.index = pd.to_datetime(time_series.index)

train_series = time_series[:'2017']
test_series = time_series['2017':]

# train_series_log = np.log(train_series)
decomposition = show_decomposition(train_series)

autocorrelation_plot(train_series)
plt.show()

