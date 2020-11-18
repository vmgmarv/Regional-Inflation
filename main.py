#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:06:53 2020

@author: marvin-corp
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
import math
import matplotlib.dates as md
import warnings
import error
warnings.filterwarnings("ignore")


region = 'r1'
#df = pd.read_csv('/home/marvin-corp/Documents/regional_inflation_forecasting_main/region_data/{}_cpi.csv'.format(region))


def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d


def read_data(region):
    
    df_inf = pd.read_csv('main_data.csv')
    
#    df_inf = pd.read_csv('/home/marvin-corp/Documents/regional_inflation_forecasting_main/region_data/{}_cpi.csv'.format(region))
    
#    df_inf = df_inf.dropna()
    df_inf.rename(columns={'Unnamed: 0': 'Year', 'Unnamed: 1': 'Month',
                       'Unnamed: 2': 'Month_number'}, inplace = True)
#        
    df_inf[region] = ((df_inf[region].shift(-12) - df_inf[region]) / df_inf[region]) * 100
    df_inf[region] = df_inf[region].shift(12)
    
    df_inf['Shocks'] = np.where((df_inf[region] - df_inf[region].shift(1) > 0),1,0)
    
    dates = month_iterator('1994-01-01', '2020-11-01')
    
    df_inf['Dates'] = dates
    df_inf['MonthYear'] = df_inf.Dates.dt.strftime('%b %Y')
    df_inf = df_inf.dropna()
    
    df_inf['Dates2'] = df_inf['Dates'].map(md.date2num) ##### datetime to integer

    return df_inf


def abs_error(df_test):
    
    df_test['abs_error'] = abs(df_test.Actual.values - df_test.Predicted.values)
    
    return df_test.abs_error.values

def normalize_(array):
   
    scaler = MinMaxScaler(feature_range = (0,1))
    inf = scaler.fit_transform(array.reshape(-1,1))
    
    return inf, scaler

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

def split_data(inf, m=2):

    X, y = split_sequence(inf, m)
    #
    X = np.array([entry[0] for entry in X]).reshape(-1,1)
    y = np.array([entry[0] for entry in y]).reshape(-1,1)
    #
    X_train = X[:math.ceil(len(X)*0.80)]
    y_train = y[:math.ceil(len(y)*0.80)]
    X_test = X[math.ceil(len(X)*0.80):]
    y_test = y[math.ceil(len(X)*0.80):]    
    
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
    
    model = input('model (svm_uni or svm_mul): ')
    
    w_regions = list(input("Region/s to forecast (type ALL to forecast all regions): ").split())
    if w_regions == 'ALL':
        regions = ['ncr', 'car', 'r1', 'r2', 'r3', 'r4a', 'r4b', 'r5',
               'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'armm']

    regions = w_regions
    
    pred_regions = []
    mae = []
    ############################################################# main
    for region in regions:
        df_inf = read_data(region)
    
        ########## features
        inf = df_inf[region].to_numpy()
        dates = df_inf.Dates.to_numpy()
        shocks = df_inf.Shocks.to_numpy()
         
        ########## Convert to 1d vector
        inf = np.reshape(inf, (len(inf),1))
        dates = np.reshape(dates, (len(dates),1))
        shocks = np.reshape(shocks, (len(shocks),1))
        
        ########## Normalize
        inf, scaler = normalize_(inf)
        
        ##########
        X_train, y_train, X_test, y_test = split_data(inf)
        X_d_tr, y_d_tr, X_d_ts, y_d_ts = split_data(dates) 
        X_s_tr, y_s_tr, X_s_ts, y_s_ts = split_data(shocks)
        
        if model == 'svm_uni':
            print('*** SVM uni ***')
            ########## Model
            clf = svm_model(X_train, y_train)
        
            m = 1 ## days
            predicted = []
            
            to_pred = np.array([inf[-1]]).reshape(1,1)
            
            for i in np.arange(1, m+1, 1):
                pred = clf.predict(to_pred)
                    
                predicted.append(pred[0])
                to_pred = pred.reshape(1,1)
            
        if model == 'svm_mul':
            print('*** SVM mul ***')
            ########## combine array to single input array
            input_x = np.concatenate((X_train, X_s_tr), axis=1)
            input_x_test = np.concatenate((X_test, X_s_ts), axis=1)
            
            ###### model
            clf = svm_model(input_x, y_train)
    
            m = 1
            predicted = []
            
            to_inf = np.array([inf[-1]]).reshape(1,1)
            to_s = np.array([shocks[-1]]).reshape(1,1)
            to_pred = np.concatenate((to_inf, to_s), axis=1)
            
            for i in np.arange(1, m+1, 1):
                pred = clf.predict(to_pred)
                
                predicted.append(pred[0])
                
                ############################################# Update values
                to_inf = np.array([pred[0]]).reshape(1,1)
                to_s = np.array([1]).reshape(1,1) ################# set shocks to 1
        #        to_s = np.array([shocks[-1]]).reshape(1,1)
                to_pred = np.concatenate((to_inf, to_s), axis=1)
            
        
            
            
        predicted = np.array(predicted)
        predicted = scaler.inverse_transform(np.array(predicted.reshape(-1,1)))
        
        print('{} forecast = {}'.format(region, round(predicted[0][0], 1)))
        pred_regions.append(round(predicted[0][0],1))
        mae.append(error.compute_mae(region, model))
        
    df_result = pd.DataFrame({'Region':regions,'Forecast':pred_regions, 'MAE as of {}'.format(error.get_last_month()):mae})
    
    df_result.to_csv('results.csv')
    
    