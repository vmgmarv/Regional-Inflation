#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:47:53 2020

@author: marvin-corp
"""

import svm_uni_opt as svm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import warnings
warnings.filterwarnings("ignore")

print("Running script")

if __name__ == '__main__':
    
    #############################################################
    df_inf = svm.read_data()
    
    ########## features
    inf = df_inf.yoy_ALL.as_matrix()
    dates = df_inf.Dates.as_matrix()
    
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    
    ########## Normalize
    inf_n, scaler = svm.normalize_(inf)
    
    ########## Remove data for testing
    m = 12 ## number of data points to be removed for testing

    train = inf_n[:(len(inf_n)-m)]    
    test = inf_n[(len(inf_n)-m):]
    t_dates = dates[(len(inf_n)-m):]
    n_inputs = inf_n[(len(inf_n)-(m+1)):-1]
        
    ########## Predict temporary
    
    predicted = []
    
    for i,j in zip(n_inputs,np.arange(1,len(n_inputs)+1,1)):
        print(".........Predicting {}/{}".format(j, m))
        ########## Split data
        X_train, y_train, X_test, y_test = svm.split_data(train)
        
        ########## Model
        clf = svm.svm_model(X_train, y_train)
        
        ########## Predict
        predicted.append(clf.predict(i.reshape(-1,1))[0])
        
        ########## Append data to train set
        train = np.append(train, clf.predict(i.reshape(-1,1))[0]).reshape(-1,1)
    
    predicted = np.array([predicted]).reshape(-1,1)
    
    ########## RMSE
    rmse = np.sqrt(np.sum(np.power(test - predicted, 2))/float(len(test)))
          
    ########## Revert data
    test_r = scaler.inverse_transform(np.array(test.reshape(-1,1)))
    predicted_r = scaler.inverse_transform(np.array(predicted.reshape(-1,1)))
    
    ########## PLot
    fig1 = plt.figure(figsize=(16,9))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t_dates, test_r, marker = 'o', label = "Out of sample inflation ({}-month data)".format(m))
    ax1.plot(t_dates, predicted_r, marker = 'o', label = "Predicted")
    ax1.xaxis.set_major_formatter(md.DateFormatter("%b'%y"))
    plt.title('NCR - SVR Univariate (RMSE: {})'.format(round(rmse,2)), fontsize = 25)
    plt.ylim(-5, 15)
    plt.ylabel('Inflation', fontsize = 18)
    plt.legend(loc = 'upper right', fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    
    
    
    
    
    
    
    
    
    
    
    