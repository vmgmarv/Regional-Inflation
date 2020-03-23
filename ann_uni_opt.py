# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:49:27 2020

@author: GABRIELVC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as md
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as MAE
import pickle
from sklearn.preprocessing import MinMaxScaler
import ann_rs_2 as ann
import math
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


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
    
    df_inf['Dates2'] = df_inf['Dates'].map(mdates.date2num) ##### datetime to integer

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

def GetOptimalCLF(train_x, train_y, rand_starts = 8):
    
    min_loss = 1e10
    
    for i in range(rand_starts):
        
        n_input = train_x.shape[1]
        
        print("Iteration number {}".format(i+1))
        
        clf = MLPRegressor(hidden_layer_sizes = (int(round(2*np.sqrt(n_input),0)),2), 
                           activation = 'tanh', solver = 'sgd', learning_rate = 'adaptive', 
                           max_iter = 10000000000, tol =  1e-10, early_stopping = True,
                           validation_fraction = 1/3)
        
        clf.fit(train_x,train_y)
        
        cur_loss = clf.loss_
        
        if cur_loss < min_loss:
            
            min_loss = cur_loss
            max_clf = clf
        
        print("Current loss {}".format(min_loss))
        
    return max_clf

if __name__ == "__main__":
    
    df_inf = read_data()
    ########## features
    inf = df_inf.yoy_ALL.as_matrix()
    dates = df_inf.Dates.as_matrix()
    
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    
    ########## Normalize
    inf, scaler = normalize_(inf)
    
    ########## Split data
    X_train, y_train, X_test, y_test = split_data(inf)
    X_d_tr, y_d_tr, X_d_ts, y_d_ts = split_data(dates) 
    
    ########## ANN proper
    clf = GetOptimalCLF(X_train, y_train)
    
    ########## Predict
    #pred_train = clf.predict(train_x)
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
    plt.title('NCR - ANN Univariate (RMSE: {})'.format(round(rmse,2)), fontsize = 25)
    plt.ylim(-5, 15)
    plt.ylabel('Inflation', fontsize = 18)
    plt.axvspan(df_test.Dates.values[0],df_test.Dates.values[-1], color = 'gray', alpha = 0.8, label = 'Forecast region')
    plt.legend(loc = 'upper right', fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=15)
