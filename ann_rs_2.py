# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:33:58 2020

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
import time

start_time = time.time()

def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d

def inf(df):
    ########################### month on month
    col = df.columns
    df = df.drop([col[0], col[1], col[2], col[3]], axis = 1)
    new_c = df.columns
    
    for i in range(len(new_c)):
        col_name = 'mom_'+ new_c[i]        
        df[col_name] = ((df[new_c[i]].shift(-1) - df[new_c[i]]) / df[new_c[i]]) * 100
        
    ########################### year on year
    for j in range(len(new_c)):
        col_name = 'yoy_' + new_c[j]
        
        df[col_name] = ((df[new_c[j]].shift(-12) - df[new_c[j]]) / df[new_c[j]]) * 100
    return df

def date_converter(df): ###for oil
    dates = []
    for i in range(len(df)):
        d =  np.datetime64(datetime.strptime(df.Variable[i], '%m-%d-%Y').date())
        d = pd.to_datetime(d)
        dates.append(d.date())
        
    return dates


def normalize(raw):
    ### raw = array
#    raw = raw[~np.isnan(raw)]
    norm = (raw - min(raw)) / (max(raw) - min(raw))
    
    return norm


def getfeatures(norm, m, train_ratio):
    
    x = []
    y = []
    
    for i in range(len(norm) - m):
        x.append(norm[i:i+m])
        y.append(norm[i+m])
    
    x = np.array(x)
    y = np.array(y)


    #### Get last index of training set

    last_index = int(len(x)*train_ratio)

    

    #### Split into training set and test set

    train_x = x[0:last_index]

    train_y = y[0:last_index]

    test_x = x[last_index:]

    test_y = y[last_index:]

    

    return train_x, train_y, test_x, test_y


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


if __name__ == '__main__':
    
    print("###############################")
    print("Script running......")
    print("###############################")
    ########################################################################### load datasets
    ######df inflation
    df_inf = pd.read_csv('ncr_cpi.csv')
    df_inf = df_inf.dropna()
    df_inf.rename(columns={'Unnamed: 0': 'Year', 'Unnamed: 1': 'Month',
                       'Unnamed: 2': 'Month_number'}, inplace = True)
        
        
    df_inf['mom_ALL'] = ((df_inf.ALL.shift(-1) - df_inf.ALL) / df_inf.ALL) * 100
    df_inf['mom_ALL'] = df_inf.mom_ALL.shift(1)
    
    
    dates = month_iterator('1994-01-01', '2020-01-30')
    df_inf['Dates'] = dates
    df_inf['MonthYear'] = df_inf.Dates.dt.strftime('%b %Y')
    df_inf = df_inf.dropna()
    
    ####################################################### adjust dates
#    start = '2001-01-01'
#    df_inf = df_inf.loc[(df_inf.Dates >= start)].reset_index()
    
    ########################################################################### normalized features
    
    norm_all = normalize(np.array(df_inf.ALL))
    train_x, train_y, test_x, test_y = getfeatures(norm_all, 5, 0.80)
    
    
    ########################################################################### ANN proper
    clf = GetOptimalCLF(train_x, train_y)
    
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    
    t_values = df_inf.ALL.values
    dn_actual = np.concatenate((train_y, test_y)) * (max(t_values) - min(t_values)) + min(t_values)
    dn_pred = np.concatenate((pred_train, pred_test)) * (max(t_values) - min(t_values)) + min(t_values)
    
    
    ########################################################################### RMSE    
    rmse_test = np.sqrt(np.sum(np.power(test_y - pred_test, 2))/float(len(test_y)))
    rmse_train = np.sqrt(np.sum(np.power(train_y - pred_train, 2))/float(len(train_y)))
#    print('RMSE_test', rmse_test)
#    print('RMSE_train', rmse_train)
    
    ########################################################################### MAE
    mae_test = MAE(test_y, pred_test)
    mae_train = MAE(train_y, pred_train)   
    
    ########################################################################### plot
    plt.plot(dn_actual, label = 'actual', color = 'blue')
    plt.plot(dn_pred, label = 'predicted', color = 'orange')
    plt.legend()
    
    #### save model
#    filename = 'ncr_rs_mlp.sav'
#    pickle.dump(clf, open(filename, 'wb'))
#    
#    
#    ### load model
#    loaded_model = pickle.load(open(filename, 'rb'))
#    pred_train = loaded_model.predict(train_x)
#    pred_test = loaded_model.predict(test_x)
    

    print("Runtime: --- %s seconds ---" % (time.time() - start_time))




