#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:34:31 2020

@author: marvin-corp
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error

def compute_mae(region, model):

    df = pd.read_excel('summary.xlsx', sheet_name=model)
    
    test_list = df.loc[df['Region'] == region].values.flatten().tolist()
    test_list = test_list[1:]
    
    pred = [test_list[i] for i in range(len(test_list)) if i % 2 != 0] 
    act = [test_list[i] for i in range(len(test_list)) if i % 2 == 0]


    mae = round(mean_absolute_error(act, pred),1)
    
    return mae

def get_last_month():
    
    df = pd.read_excel('summary.xlsx', sheet_name='svm_uni')

    lst_month = df.columns[-1]
    lst_month = lst_month[0:3]
    
    return lst_month

if __name__ == '__main__':
    
    w_regions = list(input('Region (to select Philippines, type phil): ').split())
    
    for region in w_regions:
        mae = compute_mae(region, 'svm_uni')
        if region == 'phil':
            print('Mean absolute error for Philippines: {}'.format(mae))
        else:
            print('Mean absolute error for region {}: {}'.format(region[-1],mae))
    
    lst_month = get_last_month()