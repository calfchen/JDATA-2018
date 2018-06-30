#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:35:21 2018

@author: chenxl
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

num_round=200
early_stopping_rounds=100

params_1 = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'colsample_bylevel': 0.5,
          'learning_rate': 0.1,
          'scale_pos_weight': 20,
          'seed': 666,
          'nthread': 4,
          'silent': True,
          }

params_2 = {
          'boosting_type': 'gbdt',
          'objective': 'regression_l2',
          'metric': 'mse',
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'colsample_bylevel': 0.5,
          'learning_rate': 0.1,
          'scale_pos_weight': 20,
          'seed': 666,
          'nthread': 4,
          'silent': True,
          }




































