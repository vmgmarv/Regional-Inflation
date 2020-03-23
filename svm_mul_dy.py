#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:31:40 2020

@author: marvin-corp
"""

import svm_uni_opt as svm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import warnings
warnings.filterwarnings("ignore")

#############################################################
if __name__ == '__main__':
    df_inf = svm.read_data()
    
    ########## features
    inf = df_inf.yoy_ALL.as_matrix()
    dates = df_inf.Dates.as_matrix()
    shocks = df_inf.Shocks.as_matrix()
    
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    shocks = np.reshape(shocks, (len(shocks),1))
    
    ########## Normalize
    inf_n, scaler = svm.normalize_(inf)

    ########## Remove data for testing
    m = 12 ## number of data points to be removed for testing
    
    train_inf = inf_n[:(len(inf_n)-m)]
    test_inf = inf_n[(len(inf_n)-m):]
    
    train_s = shocks[:(len(shocks)-m)]
    test_s =  shocks[(len(shocks)-m):]

    t_dates = dates[(len(inf_n)-m):]
    
    i_input = inf_n[(len(inf_n)-(m+1)):-1]
    s_input = shocks[(len(shocks)-(m+1)):-1]
    test_input = np.concatenate((i_input, s_input), axis=1)

    
    predicted = []
    
    for i in np.arange(0, m+1, 1):
        if i != m:
            print(".........Predicting {}/{}".format(i+1, m))
        
            ########## Split data
            X_train, y_train, X_test, y_test = svm.split_data(train_inf)
            X_s_tr, y_s_tr, X_s_ts, y_s_ts = svm.split_data(train_s)
            
            ########## combine array to single input array
            input_x = np.concatenate((X_train, X_s_tr), axis=1)
            input_x_test = np.concatenate((X_test, X_s_ts), axis=1)
            
            ########## Model
            clf = svm.svm_model(input_x, y_train)
            
            ########## Predict
            to_pred = test_input[i].reshape(1,2)
            
            predicted.append(clf.predict(to_pred)[0]) ####
            
            ########## Apped data to train set
#            train_inf = np.append(train_inf, test_inf[i]).reshape(-1,1) # use actual data to predict next
            train_inf = np.append(train_inf, np.array(clf.predict(to_pred)[0])).reshape(-1,1) # use predicted data to predict next
            train_s = np.append(train_s, test_s[i]).reshape(-1,1)
    
    predicted = np.array([predicted]).reshape(-1,1)

    ########## RMSE
    rmse = np.sqrt(np.sum(np.power(test_inf - predicted, 2))/float(len(test_inf)))
          
    ########## Revert data
    test_r = scaler.inverse_transform(np.array(test_inf.reshape(-1,1)))
    predicted_r = scaler.inverse_transform(np.array(predicted.reshape(-1,1)))

    ########## PLot
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t_dates, test_r, marker = 'o', label = "Out of sample inflation ({}-month data)".format(m))
    ax1.plot(t_dates, predicted_r, marker = 'o', label = "Predicted")
    ax1.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))
    plt.title('NCR - SVR Multivariate (RMSE: {})'.format(round(rmse,2)), fontsize = 25)
    plt.ylim(-5, 15)
    plt.ylabel('Inflation', fontsize = 18)
    plt.legend(loc = 'upper right', fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=15)