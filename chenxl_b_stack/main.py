#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:34:44 2018

@author: chenxl
"""

import numpy as np 
import pandas as pd 
import datetime
#from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

from feat import *
from model import *


import warnings
warnings.filterwarnings('ignore')


def day_to_date(x, line=1):    
    if x<=1:
        day = int(5)
    elif x>1 and x<30:
        day = int(x)
    else:
        day = int(14)
    if line==1:
        return '2017-09-'+str(day).zfill(2)
    else:
        return '2017-08-'+str(day).zfill(2)
    
    

offline = 0
online = 1
line = 1

if __name__=="__main__":
    print(datetime.datetime.now())
    # read data
    df_order,df_action = read_data('../data_b/')
    
    # select target of 30 or 101
    df_order_target = df_order[(df_order['cate']==101)|(df_order['cate']==30)]
    df_action_target = df_action[(df_action['cate']==101)|(df_action['cate']==30)]

    int_columns = ['buy','day','sep_ave_day','look_days','look_total_times',\
                   'seq_look_ave_day','last_look_days','is_attention',\
                   'lm_is_buy','lm_order_times','lm_sku_total_num','lm_sku_id_num']
    test_int_columns = [f for f in int_columns if f not in ['buy','day']]
    
    if line==online:
        # create train
        #begin_date = '2016-07-31'
        #end_date = '2016-12-31'
        begin_date = '2016-11-30'
        end_date = '2017-04-30'
        
        df_train = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        df_train.fillna(0, inplace=True)
        df_train[int_columns] = df_train[int_columns].astype(np.int)
        
        # select feature and categorical_features
        feature = [f for f in df_train.columns if f not in ['buy', 'day', 'user_id']]
        categorical_features = ['user_lv_cd']
        
        #train_matrix_1=lgb.Dataset(df_train[feature],label=df_train['buy'])
        #train_matrix_2=lgb.Dataset(df_train[feature],label=df_train['day'])
        
        # create valid
        #begin_date = '2017-01-31'
        #end_date = '2017-01-31'
        begin_date = '2017-05-31'
        end_date = '2017-05-31'
    
        df_valid = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        df_valid.fillna(0, inplace=True)
        df_valid[int_columns] = df_valid[int_columns].astype(np.int)
        
        valid_matrix_1=lgb.Dataset(df_valid[feature],label=df_valid['buy'])
        valid_matrix_2=lgb.Dataset(df_valid[feature],label=df_valid['day'])
        
        df_train = pd.concat([df_train,df_valid])
        train_matrix_1 = lgb.Dataset(df_train[feature],label=df_train['buy'])
        train_matrix_2 = lgb.Dataset(df_train[feature],label=df_train['day'])
        
        
        # online test
        #begin_date = '2017-02-28'
        #end_date = '2017-02-28'
        begin_date = '2017-06-30'
        end_date = '2017-06-30'
    
    
        online_test = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        online_test.fillna(0, inplace=True)
        online_test[int_columns] = online_test[int_columns].astype(np.int)
        
        # train model
        """
        model_1 = lgb.train(params_1, train_matrix_1, num_round, 
                      valid_sets=valid_matrix_1,
                      categorical_feature=categorical_features,
                      early_stopping_rounds=early_stopping_rounds)
        """
        model = SBBTree(params=params_1, stacking_num=5, bagging_num=3,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
        X = df_train[feature].values 
        y = df_train['buy'].values 

        X_pred = online_test[feature].values 
        model.fit(X,y)
        online_test["preds"] = model.predict(X_pred)
        
        """
        online_test["preds"] = model_1.predict(online_test[feature],
                       num_iteration=model_1.best_iteration).reshape((online_test.shape[0], 1))
        """        
        
        model_2 = lgb.train(params_2, train_matrix_2, num_round, 
              valid_sets=valid_matrix_2,
              categorical_feature=categorical_features,
              early_stopping_rounds=early_stopping_rounds)
        
        # predict online
        online_test["day"] = model_2.predict(online_test[feature],
                       num_iteration=model_2.best_iteration).reshape((online_test.shape[0], 1))
        
        # output file
        online_test = online_test.sort_values(by=['preds'], ascending = False)           
        online_test['pred_date'] = online_test['day'].map(lambda x: day_to_date(x))        
        online_test[['user_id','pred_date']][:50000].to_csv("../out/lgb_6_27_B_2_stack_submit.csv", index=None)
        online_test[['user_id','pred_date','preds']].to_csv("../out/lgb_6_27_B_2_stack.csv", index=None)

    else:
        # create train
        #begin_date = '2016-07-31'
        #end_date = '2016-11-30'
        begin_date = '2016-11-30'
        end_date = '2017-03-31'
        
    
        df_train = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        df_train.fillna(0, inplace=True)
        df_train[int_columns] = df_train[int_columns].astype(np.int)
        
        # select feature and categorical_features
        feature = [f for f in df_train.columns if f not in ['buy', 'day', 'user_id']]
        categorical_features = ['user_lv_cd']
        
        train_matrix_1=lgb.Dataset(df_train[feature],label=df_train['buy'])
        train_matrix_2=lgb.Dataset(df_train[feature],label=df_train['day'])
        
        # create valid
        #begin_date = '2016-12-31'
        #end_date = '2016-12-31'
        begin_date = '2017-04-30'
        end_date = '2017-04-30'
    
        df_valid = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        df_valid.fillna(0, inplace=True)
        df_valid[int_columns] = df_valid[int_columns].astype(np.int)
        
        valid_matrix_1=lgb.Dataset(df_valid[feature],label=df_valid['buy'])
        valid_matrix_2=lgb.Dataset(df_valid[feature],label=df_valid['day'])
        
        # offline test
        #begin_date = '2017-01-31'
        #end_date = '2017-01-31'
        begin_date = '2017-05-31'
        end_date = '2017-05-31'
    
        offline_test = create_feat(df_order_target,df_action_target,begin_date,end_date,is_train=1)
        offline_test.fillna(0, inplace=True)
        offline_test[test_int_columns] = offline_test[test_int_columns].astype(np.int)
       
        # train model
        model_1 = lgb.train(params_1, train_matrix_1, num_round, 
                      valid_sets=valid_matrix_1,
                      categorical_feature=categorical_features,
                      early_stopping_rounds=early_stopping_rounds)
        
        model_2 = lgb.train(params_2, train_matrix_2, num_round, 
              valid_sets=valid_matrix_2,
              categorical_feature=categorical_features,
              early_stopping_rounds=early_stopping_rounds)
        
        # predict offline
        offline_test["preds"] = model_1.predict(offline_test[feature],
                       num_iteration=model_1.best_iteration).reshape((offline_test.shape[0], 1))
        offline_test["day"] = model_2.predict(offline_test[feature],
                       num_iteration=model_2.best_iteration).reshape((offline_test.shape[0], 1))
        
        
        offline_test = offline_test.sort_values(by=['preds'], ascending = False)   
        offline_test['pred_date'] = offline_test['day'].map(lambda x: day_to_date(x,line=0)) 
        
        # compute score
        real_4_month = df_order_target[df_order_target['month'] == 4]
        real = find_fist_buy_day(real_4_month)        
        real5w = real[:50000]
        
        pred5w = offline_test[['user_id','pred_date']][:50000]
        score(pred5w, real5w)
        
        
    # feature importance
    lgb.plot_importance(model_2)
    print(datetime.datetime.now())

    
    
    

"""
[267]   valid_0's auc: 0.645016
[268]   valid_0's auc: 0.645001
Early stopping, best iteration is:
[168]   valid_0's auc: 0.647932

[149]   valid_0's l2: 76.6519
Early stopping, best iteration is:
[49]    valid_0's l2: 76.2805
2018-06-24 23:03:19.319995
"""    
    
    
    
    
    
    
    






