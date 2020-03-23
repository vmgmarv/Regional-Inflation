# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 08:37:07 2020

@author: GABRIELVC
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import ann_rs_2 as ann
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.dates as md
import warnings
warnings.filterwarnings("ignore")

print("Running script")

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def read_data():
    
    df_inf = pd.read_csv('ncr_cpi.csv')
    
    df_inf = df_inf.dropna()
    df_inf.rename(columns={'Unnamed: 0': 'Year', 'Unnamed: 1': 'Month',
                       'Unnamed: 2': 'Month_number'}, inplace = True)
        
    df_inf['yoy_ALL'] = ((df_inf.ALL.shift(-12) - df_inf.ALL) / df_inf.ALL) * 100
    df_inf['yoy_ALL'] = df_inf.yoy_ALL.shift(12)
    
    df_inf['Shocks'] = np.where((df_inf['yoy_ALL'] - df_inf['yoy_ALL'].shift(1) > 0),1,0)
    
    dates = ann.month_iterator('1994-01-01', '2020-01-30')
    
    df_inf['Dates'] = dates
    df_inf['MonthYear'] = df_inf.Dates.dt.strftime('%b %Y')
    df_inf = df_inf.dropna()
    
    df_inf['Dates2'] = df_inf['Dates'].map(md.date2num) ##### datetime to integer

    return df_inf

def normalize_(array):
   
    scaler = MinMaxScaler(feature_range = (0,1))
    inf = scaler.fit_transform(array.reshape(-1,1))
    
    return inf, scaler

def split_data(inf, m=1):

    X, y = split_sequence(inf, m)
    #
    X = np.array([entry[0] for entry in X]).reshape(-1,1)
    y = np.array([entry[0] for entry in y]).reshape(-1,1)
    #
    X_train = X[:math.ceil(len(X)*0.75)]
    y_train = y[:math.ceil(len(y)*0.75)]
    X_test = X[math.ceil(len(X)*0.75):]
    y_test = y[math.ceil(len(X)*0.75):]    
    
    return X_train, y_train, X_test, y_test


def svm_model(X_train, y_train):
    
    param_grid = {"C": np.linspace(10**(-2),10**3,100),
             'gamma': np.linspace(0.0001,1,20)}

    mod = SVR(epsilon = 0.000001,kernel='rbf')
    model = GridSearchCV(estimator = mod, param_grid = param_grid,
                         n_jobs=-1,scoring = "neg_mean_squared_error",verbose = 0)
    
    best_model = model.fit(X_train, y_train.ravel())
    
    return best_model

if __name__ == '__main__':
    
    #############################################################
    df_inf = read_data()
    
    ########## features
    inf = df_inf.yoy_ALL.as_matrix()
    dates = df_inf.Dates.as_matrix()
    
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    
    ########## Normalize
    inf, scaler = normalize_(inf)
    
    ##########
    X_train, y_train, X_test, y_test = split_data(inf)
    X_d_tr, y_d_tr, X_d_ts, y_d_ts = split_data(dates) 
    
    ########## Model
    clf = svm_model(X_train, y_train)
    
    ########## Predict
    pred_test = clf.predict(X_test)
    
    ########## Dataframe
    df_test = pd.DataFrame({'Dates':y_d_ts.reshape(len(y_d_ts),),'Actual':y_test.reshape(len(y_test),),
                            'Predicted':pred_test.reshape(len(pred_test),)})
    
    df_test['Actual'] = df_test.Actual.shift(1)
    df_test = df_test.dropna()
    
    ########## RMSE
    rmse = np.sqrt(np.sum(np.power(df_test.Actual.values - df_test.Predicted.values, 2))/float(len(df_test.Actual.values)))
    print('#####', rmse, '#####')
    
    ########## Revert back
    df_test['Actual'] = scaler.inverse_transform(np.array(df_test.Actual.values.reshape(-1,1)))
    df_test['Predicted'] = scaler.inverse_transform(np.array(df_test.Predicted.values.reshape(-1,1)))
    
    
    ########## PLot
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot(111)
    ax1.plot(df_inf.Dates, df_inf.yoy_ALL.shift(1), marker = 'o', label = "Regional inflation")
    ax1.plot(df_test.Dates, df_test.Predicted, marker = 'o', label = "Predicted")
    ax1.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))
    plt.title('NCR - SVR Univariate (RMSE: {})'.format(round(rmse,2)), fontsize = 25)
    plt.ylim(-5, 15)
    plt.ylabel('Inflation', fontsize = 18)
    plt.axvspan(df_test.Dates.values[0],df_test.Dates.values[-1], color = 'gray', alpha = 0.8, label = 'Forecast region')
    plt.legend(loc = 'upper right', fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=15)