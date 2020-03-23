# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:29:11 2020

@author: GABRIELVC
"""

import svm_uni_opt as svm
import pandas as pd
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
    inf, scaler = svm.normalize_(inf)
    
    ########## Split data
    X_train, y_train, X_test, y_test = svm.split_data(inf)
    X_s_tr, y_s_tr, X_s_ts, y_s_ts = svm.split_data(inf)
    X_d_tr, y_d_tr, X_d_ts, y_d_ts = svm.split_data(dates) 
    
    ########## combine array to single input array
    input_x = np.concatenate((X_train, X_s_tr), axis=1)
    input_x_test = np.concatenate((X_test, X_s_ts), axis=1)
    
    clf = svm.svm_model(input_x, y_train)
    
    ########## Predict
    pred_test = clf.predict(input_x_test)
    
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
    plt.title('NCR - SVR Multivariate (RMSE: {})'.format(round(rmse,2)), fontsize = 25)
    plt.ylim(-5, 15)
    plt.ylabel('Inflation', fontsize = 18)
    plt.axvspan(df_test.Dates.values[0],df_test.Dates.values[-1], color = 'gray', alpha = 0.8, label = 'Forecast region')
    plt.legend(loc = 'upper right', fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=15)